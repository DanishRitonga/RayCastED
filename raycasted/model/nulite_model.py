"""NuLiteRayCast detection model — NuLite backbone with RayCast detection head.

Wraps the NuLite encoder-decoder + NP seg head + PAN + SAM conditioning into a
DetectionModel-compatible nn.Module so the existing v1 framework
(RayCastTrainer, RayCastE2ELoss, ETL, eval_pannuke) works unchanged.

The model is built in pure Python (not YAML) because the FastViT encoder emits
multi-scale tensors that the YAML builder cannot natively route.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel

from raycasted.model.blocks.head import RayCastDetect
from raycasted.model.blocks.nulite import (
    NPHead,
    NuLiteEncoderDecoder,
    PANBottomUp,
    SAMConditioner,
)


class NuLiteRayCastModel(DetectionModel):
    """FastViT + NuLite decoder + NP seg head + PAN + RayCastDetect.

    Attributes mirror the v1 RayCastDetectionModel interface so the trainer,
    loss, and validator work unchanged:
        self.model        — list ending in the RayCastDetect head (model.model[-1])
        self.stride       — tensor([4, 8, 16])
        self.nc / names / end2end / lambda_seg
    """

    def __init__(
        self,
        nc: int = 5,
        n_rays: int = 64,
        variant: str = 'fastvit_s12',
        pretrained: bool = True,
        lambda_seg: float = 1.0,
        seed_map_target: bool = False,
        gate_scale: float = 1.0,
        head_channel_scale: float = 0.5,
        head_channel_min: int = 64,
        cls_channel_scale: float = 1.0,
        cls_channel_min: int = 0,
        aux_xy: bool = False,
        dcn_in_reg_head: bool = False,
        dcn_in_cls_head: bool = False,
        hierarchical_cls: bool = True,
        hierarchical_cls_detach: bool = True,
        hierarchical_binary_threshold: float = 0.01,
        verbose: bool = True,
    ):
        # BaseModel.__init__ only — skip DetectionModel YAML parsing.
        super(DetectionModel, self).__init__()

        self.nc = nc
        self.names = {i: str(i) for i in range(nc)}
        self.lambda_seg = lambda_seg
        self.seed_map_target = seed_map_target
        self.yaml = {'nc': nc, 'channels': 3}

        self.encoder_decoder = NuLiteEncoderDecoder(variant=variant, pretrained=pretrained)
        self.decoder0 = Conv(3, 64, 3, s=1)  # NuLite decoder0: 3 -> 64 at full res
        self.np_head = NPHead(in_channels=128)
        self.pan = PANBottomUp()
        self.conditioners = nn.ModuleList(
            [SAMConditioner(128, gate_scale), SAMConditioner(256, gate_scale), SAMConditioner(512, gate_scale)]
        )
        self.detect = RayCastDetect(
            nc=nc,
            end2end=True,
            ch=(128, 256, 512),
            n_rays=n_rays,
            head_channel_scale=head_channel_scale,
            head_channel_min=head_channel_min,
            cls_channel_scale=cls_channel_scale,
            cls_channel_min=cls_channel_min,
            aux_xy=aux_xy,
            inter_scale_competition=False,
            dcn_in_reg_head=dcn_in_reg_head,
            dcn_in_cls_head=dcn_in_cls_head,
            hierarchical_cls=hierarchical_cls,
            hierarchical_cls_detach=hierarchical_cls_detach,
            hierarchical_binary_threshold=hierarchical_binary_threshold,
        )

        # strides are fixed by architecture (P2/P3/P4 = 4/8/16)
        stride = torch.tensor([4.0, 8.0, 16.0])
        self.detect.stride = stride
        self.stride = stride

        # Plain list so model.model[-1] resolves to the head and the validator's
        # `isinstance(child, (list, Sequential))` traversal reaches RayCastDetect.
        self.model = [self.encoder_decoder, self.decoder0, self.np_head, self.pan, self.conditioners, self.detect]
        self.save = []

        # Set AFTER self.model is assigned — DetectionModel.end2end setter calls set_head_attr(self.model[-1]).
        self.end2end = True

        self.detect.bias_init()

        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'NuLiteRayCastModel: {n_params / 1e6:.2f}M params ({n_trainable / 1e6:.2f}M trainable)')

    def _compute_conditioned_feats(self, x: torch.Tensor):
        """Encode/decode, then PAN + SAM conditioning. Returns ([c2,c3,c4], np_logits)."""
        d = self.encoder_decoder(x)
        x0 = self.decoder0(x)
        xt = torch.cat([x0, d['b1']], dim=1)  # (B, 128, H, W) at stride 1
        np_logits = self.np_head(xt)
        p2, p3, p4 = self.pan(d['b3'], d['b4'], d['b5'])
        c2 = self.conditioners[0](np_logits, p2)
        c3 = self.conditioners[1](np_logits, p3)
        c4 = self.conditioners[2](np_logits, p4)
        return [c2, c3, c4], np_logits

    def forward(self, x, *args, **kwargs):
        """Detection forward.

        dict input -> self.loss (training). Tensor input -> preds dict (train
        mode, with 'np' logits attached) or (y, preds) tuple (eval mode).
        """
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        cond, np_logits = self._compute_conditioned_feats(x)
        if self.training:
            preds = self.detect(cond)
            preds['np'] = np_logits
            return preds
        return self.detect(cond)

    def loss(self, batch, preds=None):
        """Detection + segmentation loss."""
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch['img'])

        preds_dict = preds[1] if isinstance(preds, (tuple, list)) else preds
        np_logits = preds_dict.get('np') if isinstance(preds_dict, dict) else None

        loss, loss_items = self.criterion(preds, batch)

        if np_logits is not None and self.lambda_seg > 0:
            if self.seed_map_target and 'seed_map' in batch:
                # InstanSeg-style: regression on normalized distance-to-boundary,
                # center-weighted (peak at nucleus center, ~0 at boundary).
                seg = F.smooth_l1_loss(torch.sigmoid(np_logits), batch['seed_map'])
            elif 'np_mask' in batch:
                seg = F.binary_cross_entropy_with_logits(np_logits, batch['np_mask'])
            else:
                seg = None
            if seg is not None:
                # Append as an extra element (not broadcast) — trainer does loss.sum()
                # for backward; a scalar added to the 7-elem tensor would count 7x.
                loss = torch.cat([loss, (self.lambda_seg * seg).reshape(1)])

        return loss, loss_items

    def fuse(self, verbose: bool = True):
        """No-op fuse.

        self.model is a plain Python list (not nn.ModuleList), so the base
        BaseModel.fuse()'s ``self.model.modules()`` would raise AttributeError.
        RayCastDetect has no Conv-BN pairs to fuse and inference already uses
        the o2o head only, so skipping fusion is safe.
        """
        return self

    def load_nulite_ckpt(self, ckpt_path: str) -> None:
        """Load NuLite-pretrained encoder/decoder weights (V2 variant)."""
        self.encoder_decoder.load_nulite_ckpt(ckpt_path)

    def freeze_encoder_decoder_np(self) -> None:
        """Freeze encoder + decoder + NP head (phase 2a)."""
        self.encoder_decoder.freeze_encoder()
        self.encoder_decoder.freeze_decoder()
        self.decoder0.requires_grad_(False)
        self.np_head.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter (phase 2b)."""
        self.requires_grad_(True)
