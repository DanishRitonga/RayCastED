# Backbone & Neck Modification Plan

## Goal

Replace all strided 3x3 Conv layers in YOLO26's backbone and neck with **DB2 DWT + 1x1 Conv** downsampling to preserve high-frequency spatial features (cell boundaries, touching-cell separation) that standard learned downsampling destroys. This targets the Detection Quality bottleneck identified in the ablation study.

## Motivation

### The DQ Bottleneck

Run 11 (best result) shows a clear gap between segmentation and detection quality:

| Metric | RayCastED (Run 11) | LSP-DETR | Gap |
|--------|-------------------|----------|-----|
| SQ | 0.746 | 0.811 | -8.0% |
| **DQ** | **0.682** | **0.810** | **-15.8%** |

The model can draw accurate polygons (SQ competitive) but misses cells (DQ lacking). Run 12 confirmed the bottleneck is in the neck/backbone feature representation, not the detection head.

### Why Strided Conv Destroys Information

Standard strided 3x3 Conv in YOLO26's backbone and PANet neck suffers from:

1. **Aliasing**: High-frequency content above the Nyquist frequency folds back as artifacts. The learned filters optimize for task loss, not spectral properties.
2. **No frequency decomposition**: A single strided conv mixes all frequency bands indiscriminately. Fine cell boundary details (high-frequency) get averaged with smooth interior (low-frequency).
3. **Fixed receptive field per layer**: Each 3x3 conv with stride-2 sees exactly a 5x5 input patch. For cells spanning 28-40px at 0.25 MPP, this is insufficient context.

### Why DWT Is Better

Daubechies-2 (DB2) DWT provides:

1. **Perfect reconstruction**: LL + LH + HL + HH can reconstruct the original exactly. No information is destroyed.
2. **Explicit frequency separation**: Each sub-band captures a distinct frequency range:
   - **LL** (Low-Low): Smooth regions, cell interiors
   - **LH** (Low-High): Vertical edges, cell boundaries
   - **HL** (High-Low): Horizontal edges, cell boundaries
   - **HH** (High-High): Diagonal detail, corners, textures
3. **Anti-aliasing by design**: DB2 has 2 vanishing moments, suppressing polynomials up to degree 1 during downsampling.
4. **Parameter-free spatial operation**: The wavelet filters are fixed mathematical functions. The only learned component is the 1x1 conv that follows.

### Supporting Evidence

| Source | Finding | Relevance |
|--------|---------|-----------|
| **WaveCNet** (Williams & Li, CVPR 2020) | Replacing max/avg pooling with DWT improves COCO detection AP with Faster R-CNN and RetinaNet | DWT downsampling proven for detection |
| **DWT-UNet** | DWT downsampling in U-Net gives +3.2% Dice on medical segmentation | High-freq preservation directly improves boundary quality |
| **LKCell** (Cui et al., 2024) | Large-kernel depthwise conv + dilated reparameterization achieves SOTA on PanNuke (mPQ 0.508) with 78.4% FLOPs reduction vs CellViT-SAM-H | Same dataset, same cell detection task — reference for C3k2_LK only |
| **Run 12b** (this project) | COCO pretrained backbone regressed all metrics (-1.9% mAP) | Pretrained weights unhelpful, custom downsampling viable |

### Why Backbone + Neck

Run 12b confirmed COCO pretrained weights do not transfer to histopathology (domain gap). Since we train from scratch, replacing backbone strided Convs with DWT gives:

- Cleaner low-pass filtering at every scale (no aliasing artifacts propagating through the feature hierarchy)
- Explicit high-frequency preservation across the entire pyramid
- No pretrained weight compatibility loss

### Why DB2 Over Haar

| Property | Haar (db1) | DB2 (db2) |
|----------|-----------|-----------|
| Filter length | 2 taps | 4 taps |
| Vanishing moments | 1 | 2 |
| Frequency separation | Poor | Better |
| Shift invariance | Bad | Better |
| Phase offset | +0.5px | +0.36px |

Haar has poor shift invariance — small spatial shifts produce inconsistent sub-band outputs. For cell detection where cells appear at arbitrary sub-pixel positions, DB2's 2 vanishing moments provide more stable responses. The +0.36px phase offset is sub-pixel and absorbed during training (see Risks section).

## Architecture

### ResoConv Module

```
Input [B, C_in, H, W]
  |
  +-- DB2 DWT (periodization mode, fixed filters, no learnable params)
  |     +-- LL [B, C_in, H/2, W/2]  (low-frequency approximation)
  |     +-- LH [B, C_in, H/2, W/2]  (horizontal detail = vertical edges)
  |     +-- HL [B, C_in, H/2, W/2]  (vertical detail = horizontal edges)
  |     +-- HH [B, C_in, H/2, W/2]  (diagonal detail = corners/textures)
  |
  +-- Concat along channel dim -> [B, 4 * C_in, H/2, W/2]
  |
  +-- 1x1 Conv (C_in * 4 -> C_out) + BN + SiLU -> [B, C_out, H/2, W/2]
```

### DB2 Filter Coefficients

| Filter | Coefficients |
|--------|-------------|
| Low-pass (scaling) `h` | `[0.4829629, 0.8365163, 0.2241439, -0.1294095]` |
| High-pass (wavelet) `g` | `[-0.1294095, -0.2241439, 0.8365163, -0.4829629]` |

### Comparison: Strided Conv vs ResoConv

| Aspect | Strided 3x3 Conv | ResoConv |
|--------|-------------------|---------|
| High-freq info | Aliased / lost | Explicitly preserved in LH, HL, HH |
| Parameters | k^2 * C_in * C_out | 1^2 * 4C_in * C_out (fewer) |
| FLOPs (C=128, H=W=64) | ~2.4M | ~1.3M (~1.5-2x cheaper) |
| Anti-aliasing | Learned (often poor) | DB2 filter (mathematically optimal) |
| Frequency decomposition | None | 4 explicit sub-bands |
| Differentiable | Yes | Yes (DWT implemented as fixed conv2d) |
| ONNX exportable | Yes | Yes (pure PyTorch ops with periodization mode) |

### C3k2_LK Module

`C3k2_LK` is a drop-in replacement for standard `C3k2` that uses large-kernel depthwise convolutions instead of standard 3x3 Bottleneck sub-blocks. It inherits the C2f split/merge structure but replaces the inner `Bottleneck(c, c, k=(3,3))` with an `LKBottleneck` that provides much larger receptive field per block.

**Reference:** UniRepLKNet (Ding et al., 2023) for the dilated reparameterization design, and LKCell (Cui et al., 2024) for the scale-adaptive kernel sizing for cell detection.

#### Motivation

YOLO26s at P2/4 has an effective receptive field of only ~16px. At 0.25 MPP, cells span 28-40px diameter (7-10 feature pixels at P2). Standard 3x3 convolutions cannot distinguish touching cells in dense TIL fields. `C3k2_LK` uses scale-adaptive kernel sizes derived from **LKCell's receptive field analysis** (arXiv:2407.18054, Section 3.1):

> A cell detection kernel should cover at least one full cell diameter in feature space.

| Scale | Stride | Cell diameter (feature px) | Min kernel K | Dilated decomposition |
|-------|--------|---------------------------|-------------|----------------------|
| P2/4  | 4      | 7-10                      | **13x13**   | 5x5 d=1 + 7x7 d=2 + 3x3 d=3 + 3x3 d=4 + 3x3 d=5 |
| P3/8  | 8      | 3.5-5                     | **9x9**     | 5x5 d=1 + 5x5 d=2 + 3x3 d=3 + 3x3 d=4 |
| P4/16 | 16     | 1.75-2.5                  | **7x7**     | 5x5 d=1 + 3x3 d=2 + 3x3 d=3 |

The formula `(n-1)r + 1 <= K` (from LKCell Section 3.2.1) ensures each dilated branch covers the same spatial extent as the main kernel, giving the optimizer multiple effective gradient paths.

#### Architecture

```
C3k2_LK(C2f):
  cv1: 1x1 Conv(c_in → 2*c)         # split into [c, c] (inherited from C2f)
  m:   n × LKBottleneck(c, c, K)     # large-kernel sub-blocks
  cv2: 1x1 Conv((2+n)*c → c_out)    # merge all branches (inherited from C2f)
```

```
LKBottleneck(c, c, K):
  |
  +-- 1x1 PW expand (c → 2c)         # channel expansion
  |
  +-- DilatedReparamDW(K)             # large-kernel depthwise conv
  |     |
  |     +-- Main:   DW Conv KxK + BN
  |     +-- Branch: DW Conv k1xk1 (d=1) + BN
  |     +-- Branch: DW Conv k2xk2 (d=2) + BN
  |     +-- Branch: DW Conv k3xk3 (d=3) + BN
  |     +-- ... (up to 5 branches for K=13)
  |     |
  |     +-- Sum all branches          # element-wise add
  |
  +-- 1x1 PW project (2c → c)        # channel compression
  |
  +-- + residual (shortcut when c1==c2)
```

#### DilatedReparamDW

The key innovation from UniRepLKNet. At training time, multiple parallel dilated depthwise conv branches provide different spatial granularities within the same receptive field. At inference time, all branches fuse into a **single KxK depthwise conv** with zero overhead:

```python
class DilatedReparamDW(nn.Module):
    """Training: parallel dilated DW branches. Inference: single fused DW conv."""
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.branches = nn.ModuleList()
        # Main branch: full-size kernel
        self.branches.append(nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        ))
        # Dilated branches: smaller kernels with dilation
        # (n-1)*r + 1 <= K, so for K=13: (5,d=1), (7,d=2), (3,d=3), (3,d=4), (3,d=5)
        ...

    def forward(self, x):
        return sum(branch(x) for branch in self.branches)

    def fuse(self):
        """Pad smaller kernels to KxK and add weights → single fused DW conv."""
        ...
```

Fusion process:
1. Fuse BN into each conv branch
2. Convert dilated kernels to non-dilated via `F.conv_transpose2d`
3. Pad smaller kernels to KxK with zeros
4. Add all kernel weights → single KxK DW conv

This is the same reparameterization used in `RepVGGDW` (already in ultralytics `block.py`), extended to support arbitrary dilated decompositions.

#### Comparison: C3k2 vs C3k2_LK

| Aspect | C3k2 (standard) | C3k2_LK |
|--------|----------------|---------|
| Sub-block | Bottleneck(c, c, k=(3,3)) | LKBottleneck(c, c, K=scale-adaptive) |
| Spatial kernel | 3x3 (fixed) | 7x7-13x13 (scale-adaptive) |
| Depthwise | No (full conv) | Yes (groups=c, much fewer params) |
| Receptive field (P2) | ~16px | ~30-40px |
| Training params | k^2 * c^2 per block | k * c + 4*c^2 per block (DW + PW) |
| Inference params | Same as training | Single fused KxK DW conv |
| FLOPs (P2, c=128) | Higher | ~2-3x per block during training, same at inference |

#### YAML Interface

`C3k2_LK` takes an extra `kernel_size` argument after the standard C3k2 args:

```yaml
# Standard C3k2 args: [c2, c3k, e, attn]
# C3k2_LK adds:      kernel_size (K)
# Full: [c2, c3k, e, attn, kernel_size]

# Backbone (e=0.25, smaller hidden dim):
- [-1, 2, C3k2_LK, [256, False, 0.25, False, 9]]   # P3: K=9

# Neck (e=0.5, default hidden dim):
- [-1, 2, C3k2_LK, [256, True, 0.5, False, 13]]      # P2: K=13
- [-1, 2, C3k2_LK, [512, True, 0.5, False, 9]]       # P3: K=9
```

Kernel size is explicitly specified per layer in YAML for full ablation control.

#### Inheritance and Registration

```python
# C3k2_LK inherits C2f's split/merge structure
class C3k2_LK(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1,
                 shortcut=True, kernel_size=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            LKBottleneck(self.c, self.c, kernel_size, shortcut)
            for _ in range(n)
        )

# Registration in builder.py:
BASE_MODULES.add(C3k2_LK)     # for c1/c2 injection
REPEAT_MODULES.add(C3k2_LK)  # for repeat count (n) injection
```

## Model Builder: Custom parse_model Replacement

### The Problem

Ultralytics' `parse_model()` (in `ultralytics/nn/tasks.py:1539`) resolves YAML config entries and constructs the model graph. Custom blocks need membership in a local `base_modules` frozenset for automatic `c1/c2` injection and width-gain scaling. This frozenset is **local** to `parse_model()` and cannot be patched from outside without fragile source manipulation.

### The Solution: Custom Model Builder (Path C)

Instead of monkey-patching `parse_model`, we replace it entirely with our own model builder. This is done by:

1. Creating a custom `RayCastDetectionModel` that subclasses `DetectionModel` but overrides `__init__` to skip `parse_model()` and call our own builder
2. Keeping `DetectionTrainer` and all its training infrastructure (LR scheduling, AMP, checkpointing, callbacks)
3. Keeping all existing custom code (loss, assigner, head, dataset, validator) untouched

### Why Not Monkey-Patch

| Approach | Pros | Cons |
|----------|------|------|
| Monkey-patch `parse_model` | Minimal changes | Fragile across ultralytics updates; can't modify local frozenset cleanly; every new custom block requires registration hacks |
| **Custom model builder** | **Full control; no hacks; every new block is just `add to set`; no ultralytics coupling on architecture** | **~150 lines of builder code (one-time cost)** |
| Full custom trainer | Cleanest | ~630 lines, lose training loop infrastructure |

### Architecture

```
RayCastTrainer.get_model()
  |
  +-- RayCastDetectionModel(cfg, ch, nc, verbose)
  |     |
  |     +-- BaseModel.__init__()          (skip DetectionModel.__init__)
  |     +-- raycasted_parse_model(yaml)   (CUSTOM BUILDER — replaces parse_model)
  |     |     |
|     |     |     +-- BASE_MODULES = {Conv, C3k2, SPPF, C2PSA, ResoConv, C3k2_LK, ...}
    |     |     +-- REPEAT_MODULES = {C3k2, C3k2_LK, ...}
  |     |     +-- For each layer in YAML:
  |     |           Resolve class via globals()[m]
  |     |           Inject c1, c2 if in BASE_MODULES
  |     |           Insert repeat count if in REPEAT_MODULES
  |     |           Attach .i, .f, .type, .np to each module
  |     |     +-- Return (nn.Sequential, save_list)
  |     |
  |     +-- Stride computation (dummy forward)
  |     +-- bias_init()
  |
  +-- Swap Detect head -> RayCastDetect (existing logic)
  +-- Wire RayCastE2ELoss (existing logic)
```

### Structural Contract

The custom builder must produce modules with the same attributes that `BaseModel._predict_once()` and other methods expect:

| Attribute | Set by | Used by |
|-----------|--------|---------|
| `m.i` | Builder | `_predict_once`, `_profile_one_layer` |
| `m.f` | Builder | `_predict_once` (layer routing) |
| `m.type` | Builder | `_profile_one_layer`, logging |
| `m.np` | Builder | Logging |
| `self.model` | Builder | All `BaseModel` methods |
| `self.save` | Builder | `_predict_once` (which layers to cache) |
| `self.model[-1]` | Builder (last YAML entry) | Stride computation, `end2end` property, `init_criterion` |

### Adding New Custom Blocks

Every new custom block (ResoConv, C3k2_LK, future DCN, BiFPN, etc.) requires only:

1. Define the class in a `.py` file
2. Import and add it to `BASE_MODULES` (or `REPEAT_MODULES`) in `builder.py`
3. Use it in YAML

```python
# builder.py
from raycasted.model.resoconv import ResoConv
from raycasted.model.lk_block import C3k2_LK

BASE_MODULES = {
    Conv, C3k2, SPPF, C2PSA, Bottleneck,
    ResoConv, C3k2_LK,  # custom blocks
}

REPEAT_MODULES = {
    C3k2, C3k2_LK,  # C3k2_LK has repeat count like C3k2
}
```

```yaml
# yolo26s-resoconv-p2p4.yaml
- [-1, 1, ResoConv, [64]]           # works automatically with c1/c2 injection
- [-1, 2, C3k2_LK, [256, False, 0.25, False, 9]]  # large-kernel in backbone
```

```yaml
# yolo26s-dwt-p2p4.yaml
- [-1, 1, ResoConv, [64]]  # works automatically with c1/c2 injection
```

No namespace injection, no frozenset patching, no source manipulation.

## YAML Config Changes

### P2-P4 DWT Config (`yolo26s-dwt-p2p4.yaml`)

Replace 8 strided Conv layers (5 backbone + 3 neck) with `ResoConv`:

```yaml
# Backbone — 5 downsampling stages
backbone:
  - [-1, 1, ResoConv, [64]]           # 0-P1/2   (was Conv [64, 3, 2])
  - [-1, 1, ResoConv, [128]]          # 1-P2/4   (was Conv [128, 3, 2])
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, ResoConv, [256]]          # 3-P3/8   (was Conv [256, 3, 2])
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, ResoConv, [512]]          # 5-P4/16  (was Conv [512, 3, 2])
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, SPPF, [512, 5, 3, True]]
  - [-1, 2, C2PSA, [512]]

# Head — top-down FPN unchanged, bottom-up PAN uses ResoConv
head:
  # Top-down (FPN) — uses Upsample, no strided conv
  - [8, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, True]]

  # Bottom-up (PAN) — ResoConv replaces strided Conv
  - [-1, 1, ResoConv, [128]]          # P2->P3   (was Conv [128, 3, 2])
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]

  - [-1, 1, ResoConv, [256]]          # P3->P4   (was Conv [256, 3, 2])
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
```

### P3-P5 DWT Config (`yolo26s-dwt.yaml`)

Standard 3-scale detection with DWT downsampling (7 replacements: 5 backbone + 2 neck):

```yaml
backbone:
  - [-1, 1, ResoConv, [64]]           # 0-P1/2
  - [-1, 1, ResoConv, [128]]          # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, ResoConv, [256]]          # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, ResoConv, [512]]          # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, ResoConv, [1024]]         # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]
  - [-1, 2, C2PSA, [1024]]

head:
  # Top-down FPN
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]

  # Bottom-up PAN — ResoConv
  - [-1, 1, ResoConv, [256]]          # P3->P4   (was Conv [256, 3, 2])
  - [[-1, 15], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, ResoConv, [512]]          # P4->P5   (was Conv [512, 3, 2])
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[17, 20, 23], 1, Detect, [nc]]
```

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `raycasted/model/builder.py` | **CREATE** | Custom `raycasted_parse_model()` with own `BASE_MODULES`/`REPEAT_MODULES` sets. Replaces ultralytics' `parse_model()`. |
| `raycasted/model/train.py` | **MODIFY** | Add `RayCastDetectionModel` subclass, update `get_model()` to use custom builder |
| `raycasted/model/resoconv.py` | **CREATE** | `ResoConv` module: DB2 DWT + concat + 1x1 Conv projection |
| `raycasted/model/lk_block.py` | **CREATE** | `C3k2_LK` module: large-kernel depthwise C3k2 variant with dilated reparameterization |
| `raycasted/model/register.py` | **MODIFY** | Simplify — remove base_modules patching, keep minimal namespace injection for `init_criterion` compatibility |
| `raycasted/cfg/yolo26s-resoconv-p2p4.yaml` | **CREATE** | P2-P4 config with all ResoConv downsampling |
| `raycasted/cfg/yolo26s-resoconv.yaml` | **CREATE** | P3-P5 config with all ResoConv downsampling (ablation baseline) |
| `pyproject.toml` | **MODIFY** | Add `pytorch_wavelets>=1.3.0` dependency |

## Implementation Steps

### Step 1: Create `raycasted/model/builder.py` (~150 lines)

Custom model builder that replaces `parse_model()`. Key components:

```python
import copy
import ast
import contextlib

import torch
from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA, Bottleneck, Concat
from ultralytics.nn.modules.conv import DWConvTranspose2d
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import initialize_weights, make_divisible

BASE_MODULES = frozenset({
    Conv, C3k2, SPPF, C2PSA, Bottleneck, Concat, DWConvTranspose2d,
    # Custom modules — add new blocks here:
    # ResoConv,
})

REPEAT_MODULES = frozenset({
    C3k2,
    # Custom repeat modules — add here:
})

def raycasted_parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.
    
    Drop-in replacement for ultralytics.nn.tasks.parse_model() with
    full control over BASE_MODULES and REPEAT_MODULES.
    """
    max_channels = float('inf')
    nc, scales = d.get('nc'), d.get('scales')
    end2end = d.get('end2end')
    reg_max = d.get('reg_max', 16)
    depth, width = d.get('depth_multiple', 1.0), d.get('width_multiple', 1.0)
    scale = d.get('scale')
    if scales and scale:
        depth, width, max_channels = scales[scale]

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

        if m in BASE_MODULES:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in REPEAT_MODULES:
                args.insert(2, n)
                n = 1

        # ... (head-specific handling for Detect, Concat, etc. — same as original)
        # ... (attach .i, .f, .type, .np to each module)

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, str(m)[8:-2]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)
```

### Step 2: Add `RayCastDetectionModel` to `train.py`

Subclass `DetectionModel`, override `__init__` to use the custom builder:

```python
from ultralytics.nn.tasks import DetectionModel
from raycasted.model.builder import raycasted_parse_model

class RayCastDetectionModel(DetectionModel):
    def __init__(self, cfg='yolo26s.yaml', ch=3, nc=None, verbose=True):
        super(DetectionModel, self).__init__()  # BaseModel.__init__ only — skip parse_model
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if self.yaml['backbone'][0][2] == 'Silence':
            self.yaml['backbone'][0][2] = 'nn.Identity'
        self.yaml['channels'] = ch
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        self.model, self.save = raycasted_parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)
        # ... stride computation and bias_init (same logic as DetectionModel.__init__)
        initialize_weights(self)
        if verbose:
            self.info()
```

Update `RayCastTrainer.get_model()` to create `RayCastDetectionModel` instead of calling `super().get_model()`.

### Step 3: Add dependency

Add `pytorch_wavelets>=1.3.0` to `pyproject.toml` `[project.dependencies]`, then run `uv sync`.

### Step 4: Create `raycasted/model/resoconv.py`

Implement the `ResoConv` module (DB2 DWT + concat + 1x1 Conv projection) and add it to `BASE_MODULES` in `builder.py`.

### Step 5: Create `raycasted/model/lk_block.py`

Implement `C3k2_LK` module (LKBottleneck + DilatedReparamDW) and add it to both `BASE_MODULES` and `REPEAT_MODULES` in `builder.py`.

### Step 6: Simplify `raycasted/model/register.py`

Remove all `base_modules` patching logic. Keep only the minimal namespace injection needed for `RayCastDetect` in `tasks.__dict__` (for `init_criterion` compatibility when the trainer accesses `DetectionModel` internals).

### Step 7: Create YAML configs

Create `yolo26s-resoconv-p2p4.yaml` and `yolo26s-resoconv.yaml` as shown in the YAML Config Changes section.

### Step 8: Lint and test

```bash
uv run ruff check . && uv run ruff format .
uv run python -c "from raycasted.model.builder import raycasted_parse_model; print('OK')"
```

Verify model construction and forward pass with both configs.

## Ablation Plan

Fully factorial design isolating each modification individually, then testing all meaningful combinations. Continued from Run 13b.

### Two Independent Variables

| Variable | What it changes | Where in YAML |
|----------|----------------|---------------|
| **ResoConv** | All strided Conv → DB2 DWT + 1x1 conv (backbone + neck downsampling) | Backbone layers 0,1,3,5,7 + Neck bottom-up layers |
| **C3k2_LK** | Standard C3k2 → large-kernel depthwise variant (separate for backbone vs neck) | Backbone C3k2 blocks (layers 2,4,6,8) and/or Neck C3k2 blocks |

### Run Schedule

| # | ResoConv | C3k2_LK Location | Baseline | Expected Impact | Priority |
|---|---------|-------------------|----------|----------------|----------|
| **14** | **ON** | Off | Run 13 | **+0.03-0.08 mAP** | **HIGH** |
| **15** | Off | **Neck only** | Run 13 | +0.02-0.05 mAP | HIGH |
| **16** | Off | **Backbone only** | Run 13 | +0.02-0.06 mAP | HIGH |
| **17** | **ON** | **Neck only** | Run 14/15 | +0.05-0.10 mAP | HIGH |
| **18** | **ON** | **Backbone only** | Run 14/16 | +0.05-0.10 mAP | HIGH |
| **19** | **ON** | **Neck + Backbone** | Run 17/18 | +0.08-0.15 mAP | HIGH |

### What Each Run Tests

| Run | Question Answered |
|-----|-------------------|
| 14 | Does DWT downsampling improve DQ by preserving high-frequency features? |
| 15 | Do large-kernel blocks in the neck (feature fusion) improve DQ? |
| 16 | Do large-kernel blocks in the backbone (feature extraction) improve DQ? |
| 17 | Does DWT + neck LK compound positively? |
| 18 | Does DWT + backbone LK compound positively? |
| 19 | Full architecture: DWT + all LK blocks — does everything combine well? |

### Run Ordering Rationale

```
          Run 13 (baseline: P2-P4 + C2PSA@P4)
         /          |           \
    Run 14       Run 15        Run 16
    DWT only    LK neck      LK backbone
        \         /    \        /
         Run 17       Run 18
         DWT+neck    DWT+backbone
              \       /
              Run 19
              Full combined
```

Each run adds exactly **one variable** on top of its baseline, ensuring clean attribution:
- Runs 14/15/16 each test a single modification against the same baseline
- Runs 17/18 each combine DWT with one C3k2_LK variant
- Run 19 combines everything

### YAML Config Per Run

```yaml
# Run 14: ResoConv only
backbone:
  - [-1, 1, ResoConv, [64]]           # DWT replaces Conv
  - [-1, 2, C3k2, [256, False, 0.25]]  # standard C3k2
  ...

# Run 15: C3k2_LK neck only (no DWT)
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # standard Conv
  - [-1, 2, C3k2, [256, False, 0.25]]  # standard C3k2 in backbone
  ...
head:
  - [-1, 2, C3k2_LK, [256, True, 9]]  # LK in neck only
  ...

# Run 16: C3k2_LK backbone only (no DWT)
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # standard Conv
  - [-1, 2, C3k2_LK, [256, False, 0.25, 9]]  # LK in backbone
  ...
head:
  - [-1, 2, C3k2, [256, True]]        # standard C3k2 in neck
  ...

# Run 17: ResoConv + C3k2_LK neck
backbone:
  - [-1, 1, ResoConv, [64]]           # DWT
  - [-1, 2, C3k2, [256, False, 0.25]]  # standard C3k2 in backbone
  ...
head:
  - [-1, 2, C3k2_LK, [256, True, 9]]  # LK in neck
  ...

# Run 18: ResoConv + C3k2_LK backbone
backbone:
  - [-1, 1, ResoConv, [64]]           # DWT
  - [-1, 2, C3k2_LK, [256, False, 0.25, 9]]  # LK in backbone
  ...
head:
  - [-1, 2, C3k2, [256, True]]        # standard C3k2 in neck
  ...

# Run 19: ResoConv + C3k2_LK everywhere
backbone:
  - [-1, 1, ResoConv, [64]]           # DWT
  - [-1, 2, C3k2_LK, [256, False, 0.25, 9]]  # LK in backbone
  ...
head:
  - [-1, 2, C3k2_LK, [256, True, 9]]  # LK in neck
  ...
```

### Interpreting Results

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| 14 > 15 > 16 | DWT is the primary driver | Runs 17-19 will show if LK adds on top |
| 15 > 14 > 16 | Neck receptive field is the bottleneck | Prioritize C3k2_LK in neck |
| 16 > 14 > 15 | Backbone receptive field is the bottleneck | Prioritize C3k2_LK in backbone |
| 14 ≈ 15 ≈ 16 | All contribute equally | Run 19 should compound all three |
| 19 >> max(14,15,16) | Strong positive synergy | Adopt full architecture |
| 19 < max(14,15,16) | Overfitting or interference | Keep best single modification |
| 17 ≈ 14 | Neck LK adds nothing on top of DWT | Skip neck LK in final config |
| 18 >> 14 | Backbone LK + DWT synergize strongly | Backbone LK is essential with DWT |
| All runs < 13 | Both modifications hurt | Revert; investigate training stability |

## Risks and Mitigations

### ONNX Export

| Risk | Impact | Mitigation |
|------|--------|------------|
| `pytorch_wavelets` not ONNX-exportable | Blocks Jetson TensorRT deployment | Implement pure-PyTorch DWT using `F.conv2d` with fixed DB2 filter weights. `periodization` mode uses `torch.roll` + `F.conv2d(stride=2)` — all standard PyTorch ops. Can vendor a `DWTForward` that uses only `torch.nn` and `torch.nn.functional`. |
| DWT output shape mismatch with IDWT | Numerical precision issues | Not using IDWT (one-way transform). Shape is deterministic: `ceil(H/2)` x `ceil(W/2)`. |

### Training Stability

| Risk | Impact | Mitigation |
|------|--------|------------|
| DB2 phase offset (+0.36px) misaligns anchor grids | Degrades detection accuracy | The offset is sub-pixel and deterministic. The 1x1 conv and subsequent C3k2 blocks learn to compensate. No manual correction needed (confirmed: WaveCNet uses DB2 in detection networks without phase correction). |
| 4x channel expansion in early layers increases memory | OOM at large batch sizes | Only affects layers 0-1 (64ch -> 256ch before projection). The 1x1 conv immediately compresses back. Monitor GPU memory in Run 14; if tight, reduce batch or use gradient checkpointing. |
| DWT provides no learnable spatial filtering | Under-performs learned strided conv | The 1x1 conv learns which frequency components to keep. This is the standard pattern in WaveCNet and DWT-UNet. If proven insufficient, add a learnable sub-band weighting layer (4-channel attention: alpha_LL, alpha_LH, alpha_HL, alpha_HH). |

### Ultralytics Compatibility

| Risk | Impact | Mitigation |
|------|--------|------------|
| `BaseModel` methods depend on `parse_model` structural contract | Forward pass breaks | Custom builder attaches `.i`, `.f`, `.type`, `.np` to every module, matching the contract exactly |
| Ultralytics `DetectionModel` changes its `__init__` signature | Our override breaks | Pin ultralytics version. The override skips `DetectionModel.__init__` and calls `BaseModel.__init__` directly, so signature changes to `DetectionModel` are irrelevant |
| Width-gain scaling logic differs from ultralytics | Channel counts mismatch | Custom builder uses the same `make_divisible(min(c2, max_channels) * width, 8)` as the original |

## References

- **LKCell**: Cui et al., "LKCell: Efficient Cell Nuclei Instance Segmentation with Large Convolution Kernels", Neurocomputing 2024. [arXiv:2407.18054](https://arxiv.org/abs/2407.18054)
- **WaveCNet**: Williams & Li, "WaveCNet: A Wavelet-based Pooling Module for Deep Convolutional Neural Networks", CVPR 2020.
- **UniRepLKNet**: Ding et al., "UniRepLKNet: A Universal Perception Large-Kernel ConvNet", arXiv 2023.
- **RepLKNet**: Ding et al., "Scaling Up Your Kernels to 31x31", CVPR 2022.
- **PolarMask**: Xie et al., "PolarMask: Single Shot Instance Segmentation with Polar Representation", CVPR 2020.
- **LSP-DETR**: Zhang et al., "Large-Supervised Pre-training via Label Supervised Pre-training for Detection Transformers", arXiv 2024.
