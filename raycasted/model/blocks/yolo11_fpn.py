"""YOLO11 backbone + FPN top-down path as a DETR feature source.

Layers 0-16 are the stock yolo11.yaml backbone + top-down FPN, so the
official ``yolo11n.pt`` weights for ``model.{0..16}`` load directly
(a ``model.`` prefix strip). Layers 17-19 add a fresh P2 extension
(Upsample -> Concat[layer2] -> C3k2). The forward replicates
``BaseModel._predict_once`` routing and exposes the three-level pyramid
``{b3 (st4), b4 (st8), b5 (st16)}``; there is no stride-1 map (b1=None).
"""

import copy
import os

import torch
import torch.nn as nn
import ultralytics
import yaml

from raycasted.model.builder import raycasted_parse_model


def yolo11_fpn_yaml(scale='n', nc=5):
    """Build the yolo11.yaml dict with a P2 extension appended to the head.

    ``scale`` is passed to the builder's ``scales`` lookup so the width /
    depth multiples apply. The head is truncated to the top-down FPN
    (``head[:6]``, global layers 11-16 = P4/P3) and extended with a fresh
    P2 path (global layers 17-19).
    """
    cfg_path = os.path.join(os.path.dirname(ultralytics.__file__), 'cfg', 'models', '11', 'yolo11.yaml')
    with open(cfg_path) as fh:
        d = yaml.safe_load(fh)
    d = copy.deepcopy(d)
    d['scale'] = scale
    d['nc'] = nc
    p2_ext = [
        [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
        [[-1, 2], 1, 'Concat', [1]],
        [-1, 1, 'C3k2', [128, False]],
    ]
    d['head'] = d['head'][:6] + p2_ext
    return d


class Yolo11FPNBackbone(nn.Module):
    """YOLO11n backbone + FPN top-down exposing b3/b4/b5 (strides 4/8/16).

    Output channels are measured at construction (``self.out_channels``),
    e.g. (32, 64, 128) for scale ``n`` with width_multiple 0.25.
    """

    def __init__(self, scale='n', nc=5, ch=3, verbose=True):
        super().__init__()
        d = yolo11_fpn_yaml(scale=scale, nc=nc)
        self.model, self.save = raycasted_parse_model(copy.deepcopy(d), ch=ch, verbose=verbose)
        self.stride = torch.tensor([4.0, 8.0, 16.0])
        with torch.no_grad():
            outs = self(torch.zeros(1, ch, 256, 256))
        self.out_channels = (outs['b3'].shape[1], outs['b4'].shape[1], outs['b5'].shape[1])

    def forward(self, x):
        """Run backbone + FPN top-down; return {b3 (st4), b4 (st8), b5 (st16), b1: None}."""
        y = []
        for m in self.model:
            f = m.f
            if f != -1:
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]
            x = m(x)
            y.append(x)
        return {'b3': y[19], 'b4': y[16], 'b5': y[13], 'b1': None}

    def load_pretrained(self, ckpt='yolo11n.pt', device='cpu'):
        """Load official yolo11n.pt weights for layers 0-16.

        Args:
            ckpt: path or model name (downloaded on first use).
            device: load target device.

        Returns:
            (missing, unexpected) from load_state_dict.
        """
        from ultralytics import YOLO

        src = YOLO(ckpt).model.state_dict()
        sd = {}
        for k, v in src.items():
            if not k.startswith('model.'):
                continue
            rest = k[len('model.'):]
            idx = int(rest.split('.')[0])
            if idx <= 16:
                sd[rest] = v
        return self.load_state_dict(sd, strict=False)

    def freeze_backbone(self):
        """Freeze all backbone + FPN parameters."""
        self.requires_grad_(False)
