# RayCastED: Comprehensive Research Findings for FCN Improvements

## The Fundamental Problem

All failed attention experiments (train31/32/34) share one root cause:

> FCN broadcasts enriched features to all 5376 anchors (98.7% bg), washing out local cls discrimination. fg/bg gap collapsed from 2.5x to 1.6-1.8x every time.

**Any approach that works must either:**
1. Make per-anchor processing spatially adaptive WITHOUT global broadcast, OR
2. Improve the training signal itself, OR
3. Operate only AFTER the 5376→100 top-K selection (mostly-fg predictions)

---

## Tier 1: High Promise — Spatially Adaptive, No Global Broadcast

### 1. DCN in Regression Head (cv2)

**Source:** DCNv2 (Zhu et al., 2019), DCNv3 (Wang et al., 2023), DCNv4 (Xiong et al., 2024)

**What it does:** Modulated deformable convolution learns per-pixel offsets for the convolution kernel, allowing each spatial position to sample from the most informative nearby locations rather than a fixed grid. DCNv2 adds per-channel modulation masks.

**Why it works for FCN:**
- Each anchor's conv kernel adapts to local object geometry — no cross-anchor mixing
- Cell boundary prediction is inherently geometric — DCN warps sampling to follow boundaries
- Dense touching cells: rigid 3x3 samples symmetrically; DCN aligns with actual boundary
- Zero-init offsets+mask → starts as standard conv (safe)
- Regression enrichment doesn't contaminate cls (bg gets zero regression loss)

**Comparison with failed attention:**

| Property | Self-Attention (train31/32/34) | DCN |
|---|---|---|
| Aggregation scope | Global (all anchors) | Local (3x3 with learned offsets) |
| Cross-anchor mixing | Yes | No |
| Effect on bg anchors | Averages fg into bg | Processed independently |
| fg/bg gap risk | **High** (2.5x→1.7x) | **Low** |

**Implementation:** Replace 2nd `Conv(64, 64, 3)` in cv2 with `DCNConv(64, 64, 3)`. Config: `dcn_in_reg_head`. ~93K params (5.7%). DCNv2 available via `torchvision.ops.DeformConv2d`. DCN in cv3 (cls head) is moderate promise — risk of bg anchors sampling from adjacent fg.

**Key risk:** AMP float16 — force offset_conv to float32. Training speed — minimal at 3x3 c=64.

---

### 2. Switchable Atrous Convolution (SAC)

**Source:** DetectoRS (Qiao et al., 2020, arXiv:2006.02334), SAC-Net (Singh & Mukherjee, 2024, arXiv:2410.05274), Virus/Cell Foci Detection (Singh & Mukherjee, 2026, arXiv:2605.22290)

**What it does:** Instead of a fixed 3x3 conv, SAC applies multiple parallel convolutions with different dilation rates and uses a learned, spatially-varying switch function to blend them. Each spatial location selects the optimal receptive field size based on local features.

**Why it works for FCN:**
- Each anchor sees features at the optimal scale — small cells get narrow RF, larger structures get wider
- Switch function is a lightweight spatial sigmoid (1-channel conv), not attention
- No information flows between distant anchors — broadcast is structurally impossible
- Directly compatible with existing Conv-based head: replace `Conv(x, c3, 3)` with `SACBlock(x, c3, rates=[1,2,3])`
- Depthwise variant (DS-SAC) keeps FLOPs minimal

**Implementation:** Replace first cls conv layer with SAC block. ~50-100 lines new code. Config: `sac_in_cls_head`, `sac_rates`.

**Key risk:** Switch function may not learn meaningful variation if cls signal is too weak.

---

### 3. Large Kernel Attention (LKA) / Visual Attention Network (VAN) Decomposition

**Source:** Visual Attention Network (Guo et al., 2022, arXiv:2202.09741), RepLKNet (Ding et al., 2022), UniRepLKNet (2024)

**What it does:** LKA decomposes self-attention into three purely-convolutional operations: depthwise conv (local) → depthwise dilated conv (mid-range) → 1x1 conv (channel mixing). Achieves effective receptive field of ~21x21 through cheap composition. Inherently spatial — each output position depends only on its local neighborhood.

**Why it works for FCN:**
- Fundamentally local — no softmax over all anchors, no global averaging
- Each anchor enriched by wider context (neighboring cells, boundaries) without seeing all 5376 others
- VAN-B2 outperforms Swin-T on COCO detection (+2.6% AP)
- RayCastED's `LargeKernelRefinementBlock` already partially implements this (for regression only)
- Add channel attention gate after decomposition → cls head selectively amplifies informative regions

**Implementation:** Add channel gate (1x1 conv + sigmoid) to existing `LargeKernelRefinementBlock`, apply to cls head path. Low effort.

**Key risk:** Dilated conv sparse sampling (2-pixel jumps) might skip narrow structures at P2 stride=4.

---

## Tier 2: Medium-High Promise — Training Signal Improvements

### 4. YOLOv10 Consistent Dual Assignment — ALREADY ADOPTED

**Source:** YOLOv10 (Wang et al., 2024, arXiv:2405.14458), DEYO (Ouyang, 2024, arXiv:2402.16370)

**What it does:** Both O2M and O2O branches use the **same matching metric formula** with the **same α and β** hyperparameters. This ensures both branches rank anchors identically, so O2O's single positive anchor is also O2M's best positive. The paper proves (Appendix A.2) this minimizes the supervision gap between branches.

**IMPORTANT CORRECTION:** YOLOv10 does **NOT** pipe O2M predictions as soft cls targets to O2O. The two losses are computed independently and simply added. "Consistency" means identical anchor ranking via same α/β, not cross-branch supervision.

**Already adopted in RayCastED:** Both branches use `RayCastAssigner` with `alpha=0.5, beta=6.0`. The consistency property is already satisfied. The O2O head still only gets 28 fg anchors/image — this is a fundamental limit of topk2=1 assignment, not a consistency issue.

**Remaining difference from YOLOv10:** YOLOv10 detaches features before feeding O2O head (`x_detach = [xi.detach() for xi in x]`), preventing sparse O2O gradient from destabilizing backbone. RayCastED currently shares features between branches without detach. This could be tried but is low-impact since bg_fg_ratio_o2o=0 already prevents backbone contamination.

---

### 5. Grid-Sampling Ray Targets at Predicted Centroid

**Source:** LSP-DETR (criterion.py:107-118)

**What it does:** Ray targets are sampled at the **model's predicted centroid**, not the GT centroid. If the model predicts a center 5px off from GT, the target rays automatically adjust to give correct distances from that predicted location.

**Why it matters:** Currently, an anchor predicting a slightly-off-center point gets doubly-penalized (wrong centroid + "wrong" rays measured from GT center). With grid-sampling, ray targets adapt to wherever the anchor predicts the center, removing the double-penalty.

**How LSP-DETR does it:**
```python
grid = src_points * 2 - 1  # predicted points → [-1, 1]
tgt_radial_distances = F.grid_sample(
    tgt_radial_distance_map,  # [B, 2*R, H, W] precomputed
    grid.unsqueeze(1),
    align_corners=False,
)
```

**Implementation:** Requires precomputing per-pixel radial distance map (Rust stardist library or Python reimplementation). Then in loss: `F.grid_sample(radial_distance_map, predicted_centroids)`. Medium-high effort.

---

### 6. Asymmetric Max() Bound Loss

**Source:** LSP-DETR (criterion.py:16-27, matcher.py:30-41)

**What it does:**
```python
loss_min = relu(log(min_bound) - outputs)   # penalize if pred < min
loss_max = relu(outputs - log(max_bound))   # penalize if pred > max
loss = max(loss_min, loss_max)              # worst violation per ray
```

Loss is **zero** inside the `[log(min), log(max)]` interval. `max()` (not `sum()`) takes worst violation — one bad ray produces full penalty regardless of other rays.

**How this differs from RayCastED's range_l1:**
- RayCastED: `relu(r_min - pred) + relu(pred - r_max)` → sums both, fixed ±eps tolerance
- LSP-DETR: `max(relu(log(min) - pred), relu(pred - log(max)))` → takes worst, data-driven bounds

**Implementation:** Replace or complement range_l1. If data-driven bounds aren't available, use `gt_rays * (1-eps)` and `gt_rays * (1+eps)` with `max()` instead of `sum()`. ~20 lines.

---

### 7. Data-Driven Upper/Lower Ray Bounds (Rust Stardist)

**Source:** LSP-DETR (lib/stardist/src/lib.rs:10-99)

**What it does:**
- `lower_bound_rays`: For each pixel inside a nucleus, cast rays to **that specific nucleus's** boundary (inner boundary distance)
- `upper_bound_rays`: For each pixel inside any nucleus, cast rays to **any nucleus's** boundary (union boundary = outer boundary distance)
- When `allow_overlaps=False`: upper = lower (tight fit)
- When `allow_overlaps=True`: the "tube" between inner and outer boundaries captures overlap regions

**Why it matters:** Current `range_l1` uses fixed ±10% tolerance. Data-driven bounds give exactly-correct tolerance: zero inside boundary, extending through overlap regions. For overlapping nuclei, the tube naturally widens.

**Implementation:** Rust library (`lib/stardist/`) could be integrated directly (PyO3 module). Or Python reimplementation. Output: `[2, n_rays, H, W]` map per image. Medium-high effort.

---

## Tier 3: Medium Promise — Architecture Modifications

### 8. YOLO12 Area Attention (A2C2f) in Neck

**Source:** YOLO12 (ultralytics/nn/modules/block.py:1641-1868)

**What it does:** Regional self-attention that divides feature map into `area` non-overlapping regions, computes attention independently within each region. For 32x32 map with area=4: four independent 256-token problems instead of one 1024-token problem. Includes 7x7 DWConv positional encoding.

**Why it's different from failed experiments:**
- Operates on **feature maps** (not 5376 anchor tokens)
- Standard O(N²) softmax attention within small regions (not linear attention over thousands)
- 7x7 DWConv PE preserves spatial structure
- Regional fg/bg ratio much more favorable (4x4 region in dense tissue has many fg)

**Implementation:** Replace C3k2 with A2C2f at P3 and P4 neck layers. Use area=4 at P4 (16x16), area=16 at P3 (32x32). Keep standard C3k2 at P2 (64x64 too large).

**Key risk:** Even regional attention at P2 may have bg-dominated regions. Mitigate by using A2C2f only at P3/P4.

---

### 9. C3k2 with PSABlock (attn=True) in Neck

**Source:** YOLO26 (ultralytics/cfg/models/26/yolo26.yaml:50)

**What it does:** Each bottleneck in C3k2 is followed by PSABlock (self-attention + FFN). YOLO26 uses this at P5/32 scale only.

**Why it works:** PSABlock inside neck C3k2 at P4 (16x16=256 tokens) enriches features before cls head. Cheap O(N²) at 16x16 scale. Fundamentally different from train31/32 which operated on 5376 anchor tokens.

**Implementation:** One-line config change: `C3k2, [512, True]` → `C3k2, [512, True, 0.5, True]` at P4 PAN layer. The 4th argument `True` enables attn mode.

---

### 10. YOLOv10 Grouped-Conv Light Cls Head

**Source:** YOLOv10 (ultralytics/nn/modules/head.py:1756-1774)

**What it does:** Fully-grouped convolutions force complete channel independence in spatial conv, then mix channels only in 1x1 projection:
```python
nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1))  # g=x = depthwise
nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1))
nn.Conv2d(c3, nc, 1)
```

**Assessment:** More aggressive channel independence than current YOLO11 DWConv+Conv pattern. Could try for o2o cls head specifically (the bottleneck). Risk: with c3=128 and nc=5, grouped conv may lack cross-channel interaction for complex fg/bg patterns.

---

## Tier 4: Low-Medium Promise — Augmentation & Training Tricks

### 11. Gradient Clipping at 0.1

**Source:** LSP-DETR (configs/default.yaml:12): `gradient_clip_val: 0.1`

**Why:** AGENTS.md documents `-log(piou + 1e-7)` produced gradient explosions (~10^7). Clamp fixed to 1e-4, but gradient clipping provides additional safety net. Ultralytics supports `clip_grad` natively.

**Implementation:** Trivial — 1 config line.

---

### 12. Elastic Deformation Augmentation

**Source:** LSP-DETR (configs/experiment/PanNuke.yaml:61-63), U-Net (2015)

**What it does:** Random smooth displacement fields simulate natural tissue deformation. p=0.2, sigma=25, alpha=0.5.

**Implementation:** Add `albumentations.ElasticTransform` to augmentation pipeline with empty-mask guard. Low effort.

---

### 13. Backbone Finetuning with Gradual LR Ramp

**Source:** LSP-DETR (configs/default.yaml:23-25)

**What it does:** Backbone frozen for 30 epochs, then unfrozen with 0.1x learning rate. Gradually introduces backbone gradients without destabilizing head.

**Current state:** RayCastED has freeze/unfreeze tied to Hungarian phase, but no gradual LR ramp. When unfrozen, backbone gets same LR as everything else.

**Implementation:** After backbone unfreezes, use smaller LR (0.1x or 0.01x). Ultralytics supports per-parameter-group LR. Low-medium effort.

---

### 14. Distance Transform Overlap Resolution (Eval Only)

**Source:** LSP-DETR (image_processing.py:92-132)

**What it does:** Uses Euclidean Distance Transform (EDT) to resolve overlapping instance masks. Each overlapping pixel assigned to instance with highest EDT value (closest to center). More principled than area-based priority.

**Implementation:** Add as post-rasterization step in `main/eval_pannuke.py`. Low effort (scipy.ndimage.distance_transform_edt).

---

### 15. Radial Distance Prediction in Log Space

**Source:** LSP-DETR (lsp_detr.py:308-313)

**What it does:** Predicts `log(r)` directly, converts via `.exp()` at inference. Guarantees positivity and makes relative errors scale-invariant.

**Current state:** RayCastED uses `softplus` for positivity.

**Implementation:** Replace softplus head output with log-space head. Changes head, loss, and post-processing. Medium effort.

---

### 16. Additional Augmentations (from LSP-DETR)

| Augmentation | Params | p | Effect |
|---|---|---|---|
| Superpixels | n_segments=200, p_replace=0.1 | 0.1 | Forces shape/context reliance |
| Downscale | scale=0.5 | 0.15 | Simulates low-resolution patches |
| ZoomBlur | max_factor=1.05 | 0.1 | Simulates camera zoom blur |
| GaussNoise | std_range=[0, 0.44] | 0.25 | Scanner noise robustness |

All trivial to add via albumentations.

---

## Tier 5: Not Applicable to FCN (Require Transformer Decoder)

These are listed for completeness but fundamentally require query-based decoder architecture:

| Technique | Source | Why Not Applicable |
|---|---|---|
| Sliding Tile Attention (STA) | LSP-DETR | Requires decoder with explicit query tokens |
| CayleySTRING Positional Encoding | LSP-DETR | Requires attention mechanism for RoPE |
| FeatureSampling (Deformable Cross-Attention) | LSP-DETR | IS deformable cross-attention — needs decoder |
| Look Forward Twice (Iterative Refinement) | LSP-DETR | FCN has single forward pass, no iterative layers |
| Feature Level Alternation | LSP-DETR | Requires decoder that selects feature level |
| Auxiliary Losses per Decoder Layer | LSP-DETR | FCN has single prediction head, not 6 intermediate |
| Contrastive Cls Head | YOLO-World | Open-vocabulary approach, not relevant for fixed 5-class |
| PGIM (Programmable Gradient Information) | YOLOv9 | For 100+ layer networks; RayCastED is shallow |

---

## Extended FPN (P1) — NOT RECOMMENDED

Adding a P1 at stride 2 would give 128x128 features and ~21,500 total anchors. The problem is TOO MANY bg anchors, not too few fg. This would worsen the bg ratio from 98.7% to 99.7%, making o2o cls gradient sparsity worse.

---

## Already Adopted (No Action Needed)

| Technique | Source | RayCastED Location |
|---|---|---|
| `cost_inner` (grid_sample containment) | LSP-DETR | tal.py:336-429 (soft sigmoid adaptation) |
| Focal cost in matcher | LSP-DETR | tal.py:456-462 |
| Log-space L1 ray loss | LSP-DETR | loss.py:193-214 |
| Range-based L1 with tolerance band | LSP-DETR | loss.py:785-800 |
| Weighted class-and-tissue sampler | LSP-DETR/LKCell | sampler.py |
| Backbone freeze/unfreeze | LSP-DETR | loss.py:1351-1378 |
| Cosine LR schedule | LSP-DETR | pannuke.yaml:52 |
| DWConv+Conv cls head | YOLO11 | Inherited from parent Detect class |
| C2PSA in backbone | YOLO11 | Config layer 8 (P4 bottleneck) |
| reg_max=1 + end2end=True | YOLO26 | DFL removed, dual assignment |
| Inter-scale competition | Novel | head.py:725-771 (inference softmax) |
| Consistent dual assignment (same α/β) | YOLOv10 | Both branches use RayCastAssigner(α=0.5, β=6.0) |

---

## Summary: Recommended Experiment Priority

| Priority | Experiment | Addresses | Risk | Effort |
|---|---|---|---|---|
| **1** | DCN in cv2 (reg head) | Polygon quality (bPQ, AJI) | Low | Low |
| **2** | SAC in cv3 (cls head) | Per-anchor adaptive RF | Low | Low |
| **3** | Asymmetric max() bound loss | Regression loss quality | Low | Low (~20 lines) |
| **4** | Gradient clipping at 0.1 | Training stability | Trivial | Trivial |
| ~~5~~ | ~~YOLOv10 consistent dual assignment~~ | ~~O2O cls gradient signal~~ | **Already adopted** | N/A |
| **5** | Grid-sample targets at predicted centroid | Ray target adaptation | Low | Med-High |
| **6** | A2C2f or attn=True at P4 neck | Feature enrichment | Medium | Medium |
| **7** | Elastic deformation augmentation | Data augmentation | Low | Low |
| **8** | Data-driven ray bounds (Rust stardist) | Range tolerance accuracy | Low | Med-High |
| **9** | LKA in cls head | Wider local context | Low | Low |

**Quick wins (implement in one session):** #1, #3, #4, #7
**High impact, medium effort:** #2, #5
**Architecture changes:** #6, #9
