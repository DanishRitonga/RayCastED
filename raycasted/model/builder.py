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
    DWConv,
    DWConvTranspose2d,
    HGBlock,
    HGStem,
    RepC3,
)
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.ops import make_divisible

from raycasted.model.blocks.head import RayCastDetect
from raycasted.model.blocks.dcn_blocks import C3k2_DCN
from raycasted.model.blocks.lk_block import C3k2_LK
from raycasted.model.blocks.resoconv import (
    DWT_HF,
    DWT_LL,
    HFResidual,
    ResoConv,
    ResoConvDS,
    ResoConvDS_Hybrid,
    ResoConvHybrid,
)

BASE_MODULES = frozenset(
    {
        Conv,
        DWConv,
        C3k2,
        SPPF,
        C2PSA,
        Bottleneck,
        DWConvTranspose2d,
        RepC3,
        ResoConv,
        ResoConvDS,
        ResoConvDS_Hybrid,
        ResoConvHybrid,
        C3k2_LK,
        C3k2_DCN,
    }
)

REPEAT_MODULES = frozenset(
    {
        C3k2,
        C2PSA,
        C3k2_LK,
        C3k2_DCN,
        RepC3,
    }
)

DETECT_MODULES = frozenset({Detect, RayCastDetect})


def _resolve_ch(ch_list, f):
    """Resolve output channels from a `from` index (int or singleton list)."""
    return ch_list[f] if isinstance(f, int) else ch_list[f[0]]


def _handle_dwt_ll(ch_list, f, args, layers):
    c1 = _resolve_ch(ch_list, f)
    wavelet_type = args[0] if len(args) > 0 else 'bior2.2'
    return DWT_LL(c1, wavelet_type=wavelet_type), c1


def _handle_dwt_hf(ch_list, f, args, layers):
    c1 = _resolve_ch(ch_list, f)
    drop_hh = args[0] if len(args) > 0 else False
    wavelet_type = args[1] if len(args) > 1 else 'bior2.2'
    n_hf = 2 if drop_hh else 3
    return DWT_HF(c1, drop_hh=drop_hh, wavelet_type=wavelet_type), c1 * n_hf


def _handle_hf_residual(ch_list, f, args, layers):
    source_idx = int(args[0])
    source_c2 = ch_list[source_idx]
    c2 = _resolve_ch(ch_list, f)
    m_ = HFResidual(c2, source_c2)
    m_._source = layers[source_idx]
    return m_, c2


def _handle_hgstem(ch_list, f, args, layers):
    c1 = _resolve_ch(ch_list, f)
    cm = args[0]
    c2 = args[1] if len(args) > 1 else cm
    return HGStem(c1, cm, c2), c2


def _handle_hgblock(ch_list, f, args, layers):
    c1 = _resolve_ch(ch_list, f)
    cm = args[0]
    c2 = args[1] if len(args) > 1 else cm
    k = args[2] if len(args) > 2 else 3
    lightconv = args[3] if len(args) > 3 else False
    shortcut = args[4] if len(args) > 4 else False
    return HGBlock(c1, cm, c2, k, lightconv=lightconv, shortcut=shortcut), c2


_SPECIAL_HANDLERS = {
    HGStem: _handle_hgstem,
    HGBlock: _handle_hgblock,
    DWT_LL: _handle_dwt_ll,
    DWT_HF: _handle_dwt_hf,
    HFResidual: _handle_hf_residual,
}


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
        if isinstance(m, str):
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    args[j] = ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n
        m_ = None

        if m in _SPECIAL_HANDLERS:
            m_, c2 = _SPECIAL_HANDLERS[m](ch, f, args, layers)
        elif m in BASE_MODULES:
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
