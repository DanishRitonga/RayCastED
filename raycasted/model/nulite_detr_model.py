"""NuLite-RayCast DETR detection model.

FastViT s12 encoder + NuLite decoder (seg backbone) feeding a single query-based
DETR head. Queries are seeded from local maxima of the NP seed map (InstanSeg
style self-prompting) instead of FCN multi-scale anchors. Uses the
NuLiteDETRLoss (Hungarian 1:1 matching + VFL + ray-L1 + polar-IoU).
"""

import torch
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel

from raycasted.model.blocks.nulite import NPHead, NuLiteEncoderDecoder
from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder
from raycasted.model.nulite_detr_loss import NuLiteDETRLoss


class NuLiteRayCastDETRModel(DetectionModel):
    """Single-head DETR detector on the NuLite seg backbone.

    Submodules (plain-list self.model ending in the DETR decoder so that
    model.model[-1] is the head, matching the v1 framework contract):
      encoder_decoder -> NuLiteEncoderDecoder (b3/b4/b5/b1)
      decoder0        -> Conv(3, 64, 3)
      np_head         -> NPHead(128)  (seed logits at full res)
      detr            -> NuLiteDETRDecoder (nc, ch=(64,128,256), nq=300, ndl=3)
    """

    def __init__(
        self,
        nc: int = 5,
        n_rays: int = 64,
        variant: str = 'fastvit_s12',
        pretrained: bool = True,
        lambda_seg: float = 1.0,
        nq: int = 300,
        ndl: int = 3,
        hd: int = 256,
        seed_threshold: float = 0.5,
        peak_distance: int = 3,
        grid_size: float = 0.05,
        query_selection: str = 'grid',
        seed_feature: bool = True,
        no_object: bool = True,
        seed_in_content: bool = True,
        no_object_weight: float = 1.0,
        mds: bool = True,
        verbose: bool = True,
    ):
        super(DetectionModel, self).__init__()
        self.nc = nc
        self.n_rays = n_rays
        self.lambda_seg = lambda_seg
        self.no_object = no_object
        self.no_object_weight = no_object_weight
        self.names = {i: str(i) for i in range(nc)}
        self.yaml = {'nc': nc, 'n_rays': n_rays, 'architecture': 'nulite_detr', 'variant': variant}

        self.encoder_decoder = NuLiteEncoderDecoder(variant=variant, pretrained=pretrained)
        self.decoder0 = Conv(3, 64, 3, s=1)
        self.np_head = NPHead(in_channels=128)
        self.detr = NuLiteDETRDecoder(
            nc=nc,
            ch=(64, 128, 256),
            hd=hd,
            nq=nq,
            ndl=ndl,
            nd=0,
            n_rays=n_rays,
            seed_threshold=seed_threshold,
            peak_distance=peak_distance,
            grid_size=grid_size,
            query_selection=query_selection,
            seed_feature=seed_feature,
            no_object=no_object,
            seed_in_content=seed_in_content,
            mds=mds,
        )

        stride = torch.tensor([4.0, 8.0, 16.0])
        self.detr.stride = stride
        self.stride = stride

        self.model = [self.encoder_decoder, self.decoder0, self.np_head, self.detr]
        self.save = []
        # NOTE: end2end stays False. The DetectionModel.end2end getter reads
        # model[-1].end2end (absent on the DETR decoder), and the setter warns +
        # skips when the head lacks the attr. Keeping it False also avoids the
        # ultralytics resume path calling criterion.update() (NuLiteDETRLoss has
        # no anneal method).

        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            print(f'NuLiteRayCastDETRModel: {n_params / 1e6:.2f}M params')

    # -- inference path -----------------------------------------------------

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None, *args, **kwargs):
        """Run encoder -> decoder -> seed head -> DETR decoder.

        Training: returns the decoder's 5-tuple (dec_polygons, dec_scores,
        enc_polygons, enc_scores, dn_meta).
        Eval: returns (y, out) where y = (B, nq, raycast_dim + 2) in the Detect
        format [cx, cy, d1..dn, conf, cls] (eval_pannuke / val.py compatible).
        """
        d = self.encoder_decoder(x)
        x0 = self.decoder0(x)
        xt = torch.cat([x0, d['b1']], dim=1)
        seed = self.np_head(xt)
        self._last_seed = seed

        out = self.detr([d['b3'], d['b4'], d['b5']], batch, seed_map=seed)
        if self.training:
            return out

        y, out = out
        poly = y[..., : self.detr.raycast_dim]
        scores = y[..., self.detr.raycast_dim:]  # already sigmoid prob (nc or nc+1)
        if self.no_object:
            # LSP-DETR: last column is the explicit "no object" prob.
            # conf = 1 - P(no-object); cls = argmax over the nc real classes.
            p_no_obj = scores[..., -1:]
            conf = (1.0 - p_no_obj).clamp(min=0.0)
            cls = scores[..., :-1].argmax(-1, keepdim=True)
        else:
            conf = scores.max(-1, keepdim=True).values
            cls = scores.argmax(-1, keepdim=True)
        # decode_polygon outputs NORMALIZED coords (sigmoid xy, exp rays in
        # [0,1]). val.py / eval_pannuke expect PIXEL coords (GT is denormalized
        # by crop_size there), so scale to pixels using the input size.
        imgsz = x.shape[-1]
        poly = poly.clone()
        poly[..., :2] = poly[..., :2] * imgsz
        poly[..., 2:] = poly[..., 2:] * imgsz
        y_detect = torch.cat([poly, conf, cls], -1)
        return y_detect, out

    # -- training path ------------------------------------------------------

    def init_criterion(self):
        """DETR 1:1 Hungarian criterion (no o2m/o2o dual assignment).

        Focal loss with an explicit "no object" class (LSP-DETR style) when
        ``no_object`` is enabled; ``no_object_weight`` scales the ∅ column
        (>1 = suppress redundant per-nucleus queries harder).
        """
        return NuLiteDETRLoss(
            nc=self.nc,
            loss_gain={'class': 1, 'ray': 5, 'piou': 2, 'no_object': self.no_object_weight},
            use_vfl=not self.no_object,
            no_object=self.no_object,
            n_rays=self.n_rays,
        )

    def loss(self, batch, preds=None):
        """DETR loss + seed-map seg loss.

        Args:
            batch: dict with img (B,3,H,W), batch_idx (M,), cls (M,),
                bboxes (M, 2+n_rays), seed_map (B,1,H,W).
            preds: optional pre-computed decoder output (validator path).
        """
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()

        img = batch['img']
        device = img.device
        bs = img.shape[0]
        batch_idx = batch['batch_idx'].to(device, torch.long).view(-1)
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': batch['cls'].to(device, torch.long).view(-1),
            'bboxes': batch['bboxes'].to(device),
            'batch_idx': batch_idx,
            'gt_groups': gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        else:
            preds = preds[1] if isinstance(preds, tuple) and len(preds) == 2 else preds

        dec_polygons, dec_scores, enc_polygons, enc_scores, dn_meta = preds
        if dn_meta is not None:
            dn_bs, dn_num_quad, _ = dn_meta
            dn_num_group = dn_num_quad * 2
            split = [dn_bs * dn_num_group, bs]
            dec_polygons = dec_polygons.split(split, dim=1)[1]
            dec_scores = dec_scores.split(split, dim=1)[1]
            enc_polygons = enc_polygons.split(split, dim=1)[1]
            enc_scores = enc_scores.split(split, dim=1)[1]

        dec_polygons = torch.cat([enc_polygons.unsqueeze(0), dec_polygons], dim=0)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores], dim=0)

        loss = self.criterion((dec_polygons, dec_scores), targets)
        total = sum(loss.values())

        seed = getattr(self, '_last_seed', None)
        if seed is not None and self.lambda_seg > 0 and 'seed_map' in batch:
            seg = F.l1_loss(seed, batch['seed_map'].to(device))
            total = total + self.lambda_seg * seg

        loss_items = torch.as_tensor(
            [loss['loss_class'].detach(), loss['loss_ray'].detach(), loss['loss_piou'].detach()],
            device=device,
        )
        return total, loss_items

    # -- helpers ------------------------------------------------------------

    def fuse(self, verbose=True):
        """No-op (DETR decoder has no Conv-BN pairs to fold)."""
        return self

    def load_nulite_ckpt(self, ckpt_path):
        """Load NuLite pretrained encoder/decoder weights."""
        self.encoder_decoder.load_nulite_ckpt(ckpt_path)

    def freeze_encoder_decoder_np(self):
        """Freeze encoder + decoder + decoder0 + NP seed head."""
        self.encoder_decoder.freeze_encoder()
        self.encoder_decoder.freeze_decoder()
        self.decoder0.requires_grad_(False)
        self.np_head.requires_grad_(False)

    def unfreeze_all(self):
        """Unfreeze every parameter."""
        self.requires_grad_(True)
