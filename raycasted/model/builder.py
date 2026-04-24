"""Custom model builder for RayCastED.

Replaces ultralytics.nn.tasks.parse_model() with full control over
BASE_MODULES and REPEAT_MODULES. This enables custom blocks (ResoConv,
C3k2_LK, etc.) without fragile monkey-patching of local frozensets.

Usage:
    from raycasted.model.builder import raycasted_parse_model
    from ultralytics.nn.tasks import DetectionModel
    import copy

    class RayCastDetectionModel(DetectionModel):
        def __init__(self, cfg='yolo26s.yaml', ch=3, nc=None, verbose=True):
            super(DetectionModel, self).__init__()  # BaseModel.__init__ only
            self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
            if self.yaml['backbone'][0][2] == 'Silence':
                self.yaml['backbone'][0][2] = 'nn.Identity'
            self.yaml['channels'] = ch
            if nc and nc != self.yaml['nc']:
                self.yaml['nc'] = nc
            self.model, self.save = raycasted_parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
            # ... stride computation, bias_init, etc.
"""

import ast
import contextlib

import torch
from ultralytics.nn.modules import (
    C2PSA,
    SPPF,
    Bottleneck,
    C3k2,
    Concat,
    Conv,
    DWConvTranspose2d,
)
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.ops import make_divisible

from raycasted.model.blocks.head import RayCastDetect
from raycasted.model.blocks.lk_block import C3k2_LK
from raycasted.model.blocks.resoconv import ResoConv

BASE_MODULES = frozenset(
    {
        Conv,
        C3k2,
        SPPF,
        C2PSA,
        Bottleneck,
        DWConvTranspose2d,
        ResoConv,
        C3k2_LK,
    }
)

REPEAT_MODULES = frozenset(
    {
        C3k2,
        C2PSA,
        C3k2_LK,
    }
)


def raycasted_parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Drop-in replacement for ultralytics.nn.tasks.parse_model() with
    full control over BASE_MODULES and REPEAT_MODULES. This enables
    custom blocks (ResoConv, C3k2_LK) without fragile monkey-patching.

    Args:
        d (dict): Model dictionary (from YAML).
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        (torch.nn.Sequential): PyTorch model.
        (list): Sorted list of layer indices whose outputs need to be saved.
    """
    from ultralytics.utils import LOGGER

    max_channels = float('inf')
    nc, scales = d.get('nc'), d.get('scales')
    end2end = d.get('end2end')
    reg_max = d.get('reg_max', 16)
    depth, width = d.get('depth_multiple', 1.0), d.get('width_multiple', 1.0)
    scale = d.get('scale')
    if scales and scale:
        depth, width, max_channels = scales[scale]

    if verbose:
        LOGGER.info(f'\n{"":>3}{"from":>20}{"n":>3}{"params":>10}  {"module":<45}{"arguments":<30}')

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # Resolve module class
        if isinstance(m, str):
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]

        # Evaluate string args
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n

        # Module-specific channel handling (order matters!)
        if m in BASE_MODULES:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]

            if m in REPEAT_MODULES:
                args.insert(2, n)
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is torch.nn.Upsample:
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]
        elif m in frozenset(
            {
                Detect,
                RayCastDetect,
            }
        ):
            args = [nc, reg_max, end2end, [ch[x] for x in f]]
        else:
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t

        if verbose:
            LOGGER.info(f'{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}')

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)
