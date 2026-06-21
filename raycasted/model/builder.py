"""Custom model builder for RayCastED.

Replaces ultralytics.nn.tasks.parse_model() with full control over
BASE_MODULES and REPEAT_MODULES. This enables custom blocks (ResoConv)
without fragile monkey-patching of local frozensets.
"""

import ast
import contextlib

import torch
from ultralytics.nn.modules import (
    C2PSA,
    C3k2,
    Concat,
    Conv,
    SPPF,
)
from ultralytics.utils.ops import make_divisible

from raycasted.model.blocks.dcn_blocks import C3k2_DCN
from raycasted.model.blocks.head import RayCastDetect
from raycasted.model.blocks.resoconv import ResoConv, ResoConvHybrid

BASE_MODULES = frozenset(
    {
        Conv,
        C3k2,
        SPPF,
        C2PSA,
        ResoConv,
        ResoConvHybrid,  # backward-compat alias for old yamls/checkpoints
        C3k2_DCN,
    }
)

REPEAT_MODULES = frozenset(
    {
        C3k2,
        C2PSA,
        C3k2_DCN,
    }
)

DETECT_MODULES = frozenset({RayCastDetect})


def _resolve_ch(ch_list, f):
    """Resolve output channels from a `from` index (int or singleton list)."""
    return ch_list[f] if isinstance(f, int) else ch_list[f[0]]


def raycasted_parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Drop-in replacement for ultralytics.nn.tasks.parse_model() with
    full control over BASE_MODULES and REPEAT_MODULES.

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
        if isinstance(m, str):
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    args[j] = ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n
        m_ = None

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
            c2 = _resolve_ch(ch, f)
        elif m in DETECT_MODULES:
            args = [nc, reg_max, end2end, [ch[x] for x in f]]
        else:
            c2 = _resolve_ch(ch, f)

        if m_ is None:
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

    model = torch.nn.Sequential(*layers)
    model.save = sorted(save)
    return model, model.save
