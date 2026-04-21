# Ablation Study — PanNuke Fold3

Tracking the effect of each config change on detection quality.

**Dataset:** PanNuke fold1=train, fold2=val, fold3=test
**Base model:** yolo26s, 64 rays, 256px crop, 200 epochs, batch=16, AMP=true

---

## Results

| # | Config | AJI | mAP@0.5 | mAP@0.75 | PQ | SQ | DQ | Prec | Recall | F1 |
|---|--------|-----|---------|----------|-----|-----|-----|------|--------|------|
| 1 | Baseline (no training config, 32 rays) | 0.534 | — | — | — | — | — | — | — | 0.636 |
| 2 | Baseline (no training config, 64 rays) | 0.543 | 0.377 | — | 0.545 | 0.723 | 0.708 | — | — | 0.636 |
| 3 | + all recommended (topk=20, rad=2.0, focal, wide head, augment, cos_lr) | 0.461 | 0.223 | 0.095 | 0.317 | 0.701 | 0.421 | 0.331 | 0.690 | 0.447 |
| 4 | revert assigner (topk=13, rad=1.5), keep rest | 0.530 | 0.220 | 0.119 | 0.353 | 0.738 | 0.447 | 0.324 | 0.691 | 0.442 |
| 4m | same as Run 4 but yolo26m (24M params) | 0.509 | 0.231 | 0.119 | 0.358 | 0.735 | 0.456 | 0.343 | 0.687 | 0.458 |
| 5 | revert focal loss (BCE), keep wide head + augment + cos_lr (yolo26s) | 0.542 | 0.379 | 0.144 | 0.526 | 0.724 | 0.680 | 0.681 | 0.591 | 0.633 |
| 6 | + -log(IoU) loss (PolarMask formulation), same config as Run 5 | 0.553 | 0.384 | 0.187 | 0.536 | 0.745 | 0.676 | 0.680 | 0.601 | 0.638 |
| 7 | + soft polar centerness (PolarMask++), same config as Run 6 | 0.553 | 0.389 | 0.174 | 0.538 | 0.736 | 0.686 | 0.695 | 0.598 | 0.643 |
| 8 | same as Run 7 but scale/translate augment OFF | 0.553 | 0.374 | 0.164 | 0.550 | 0.730 | 0.709 | 0.642 | 0.624 | 0.633 |
| 9 | same as Run 7 but yolo26s-p2 (4-scale, stride 4/8/16/32) | 0.538 | 0.381 | 0.135 | 0.520 | 0.722 | 0.675 | 0.684 | 0.588 | 0.632 |
| 10 | same as Run 7 but QFL (quality focal loss) replaces BCE | 0.529 | 0.243 | 0.134 | 0.379 | 0.740 | 0.480 | 0.370 | 0.669 | 0.477 |
| **11** ⭐ | **E2E Fix + Hungarian matching (tal_topk=13, beta=3.0, o2o.topk=1)** | **0.558** | **0.428** | **0.250** | **0.543** | **0.746** | **0.682** | **0.685** | **0.598** | **0.639** |
| 12 | P3-P5 + Large Kernel (refinement_kernel_size=7) | 0.561 | 0.425 | 0.248 | 0.545 | 0.749 | 0.683 | 0.686 | 0.598 | 0.639 |
| 12b | P3-P5 + Pretrained Backbone (yolo26s.pt, COCO) | ? | ? | ? | ? | ? | ? | ? | ? | ? |

---

## Config Details

### Run 1 — Baseline (32 rays)
No `training:` section in YAML. All legacy defaults.
- head: c2 = max(16, ch[0]//4) = 16 at P3
- assigner: topk=13, radius_scale=1.5
- loss: BCE, no focal
- augmentation: flip/rotate only (no stain, scale, translate)
- LR: linear decay
- n_rays: 32

### Run 2 — Baseline (64 rays)
Same as Run 1 but n_rays=64.

### Run 3 — All recommended changes
```yaml
training:
  head_channel_scale: 0.5
  head_channel_min: 64
  cos_lr: true
  assigner_topk: 20
  assigner_radius_scale: 2.0
  focal_loss: true
  focal_gamma: 2.0
  stain_jitter: true
  scale_augment: true
  translate_augment: true
```

### Run 4 — Revert assigner
Same as Run 3 but:
```yaml
  assigner_topk: 13       # reverted from 20
  assigner_radius_scale: 1.5  # reverted from 2.0
```

### Run 5 — Revert focal loss
Same as Run 4 but:
```yaml
  focal_loss: false       # reverted from true
  assigner_topk: 13       # reverted from 20
  assigner_radius_scale: 1.5  # reverted from 2.0
```
Result: full recovery to baseline. Focal loss confirmed harmful.

### Run 11 — E2E Fix + Hungarian Matching
Same as Run 7 but:
```yaml
  tal_topk: 13              # Controls one2many branch
  use_hungarian_o2o: true   # Hungarian matching for one2one branch (globally optimal)
  assigner_beta: 3.0        # Lowered from 6.0 for better dense cell handling
```
Result: mAP@0.5 +0.039, mAP@0.75 +0.076. Significant improvement at stricter thresholds.

### Run 12 — P3-P5 + Large Kernel (Phase 1) ✅ COMPLETE
Same as Run 11 but:
```yaml
  refinement_kernel_size: 7  # LargeKernelRefinementBlock in detection head
```
Result: **Negligible improvement.** mAP@0.5 -0.003 (0.428→0.425), SQ +0.003 (0.746→0.749), DQ +0.001 (0.682→0.683). Large kernel in the head does NOT address the bottleneck.

### Run 12b — P3-P5 + DWT Neck (Phase 1b) — PENDING
Replace PANet downsampling conv with Daubechies-2 DWT in the neck only.
Purpose: Test if preserving high-frequency details during feature fusion helps DQ.
Expected: +0.01-0.04 mAP if neck is the bottleneck.

### Run 12b — P3-P5 + Pretrained Backbone (COCO) 🔄 IN PROGRESS
Same as Run 11 but with COCO-pretrained backbone (layers 0-10):
```yaml
  pretrained_backbone: "yolo26s.pt"  # 5.46M backbone params from COCO
  # Neck and head remain randomly initialised
```
Purpose: Test if transfer learning from COCO backbone closes the DQ gap.
240/240 backbone keys compatible (identical architecture).
Status: Config updated, ready to train.

### Run 13 — P2-P4 Scale (Standard 3×3 Head) — PENDING
Same as Run 11 but with P2-P4 scales (P5 removed):
```yaml
global_settings:
  model: "raycasted/cfg/yolo26s-p2p4.yaml"  # P2/4, P3/8, P4/16 — no P5

training:
  refinement_kernel_size: 3  # Standard head (isolate P2 effect)
```
Purpose: Isolate P2's contribution. Run 9 tested P2-P5 (with P5 noise), this tests P2-P4 only.
7.06M params, 25.6 GFLOPs (vs 10.01M/22.8 GFLOPs baseline).

---

## Observations

- **Run 3 vs Run 2:** All recommended changes together caused mAP collapse (0.377→0.223). High recall (0.69) but very low precision (0.33) suggests the model cannot suppress false positives.
- **Run 4 vs Run 3:** Reverting assigner improved AJI (+0.07) and SQ (+0.04) but mAP stayed flat. Assigner was not the main cause of the mAP drop.
- **Run 4m vs Run 4:** Doubling backbone (s→m, 10.7M→24.0M params) gave negligible improvement (+0.011 mAP, +0.016 F1). Model capacity is not the bottleneck. The low precision (0.34) persists — problem is in loss/augmentation config, not model size.
- **Run 5 vs Run 2:** Reverting focal loss recovered all metrics to baseline. mAP@0.5: 0.379 vs 0.377 (baseline). The wide head, stain jitter, scale/translate augment, and cos_lr are neutral — not hurting, not significantly helping yet. **Conclusion: focal loss is harmful for YOLO-based polygon detection.** BCE should be the default.
- **Run 6 vs Run 5:** `-log(IoU)` loss (PolarMask formulation) improved all metrics except DQ (-0.004 noise). mAP@0.75 gained +0.043 and SQ +0.021 — stronger gradients on low-quality predictions directly improve polygon quality at stricter IoU thresholds. **Conclusion: `-log(IoU)` is superior to `1 - IoU`.**
- **Run 7 vs Run 6:** Soft polar centerness (PolarMask++) improved precision (+0.015) as expected — the centerness branch suppresses off-center duplicate predictions. However, the overall gain is marginal (+0.005 mAP@0.5, +0.005 F1) with a slight SQ regression (-0.009) and mAP@0.75 regression (-0.014). The YOLO anchor-based framework already has strong assignment (RayCastAssigner with polar IoU), so centerness provides limited additional benefit. **Conclusion: soft polar centerness is neutral-to-slightly-positive for YOLO-based polygon detection. Worth keeping but not a major quality lever.**
- **Run 8 vs Run 7:** Removing scale/translate augmentation hurt precision by 5 points (0.695→0.642) with mAP@0.5 dropping -0.015. Earlier comparison (Run 5 vs Run 2) falsely suggested augmentations were neutral — it was confounded by the wide head addition. **Conclusion: scale and translate augmentation are beneficial (+0.015 mAP, +0.053 precision). Must be kept enabled.**
- **Run 9 vs Run 7:** P2 detection scale (stride 4/8/16/32, 5440 anchors) regressed across all metrics vs 3-scale (stride 8/16/32, 1344 anchors). mAP@0.75 dropped -0.039. At 256px/0.25 MPP, cells are ~28px diameter — already well-covered by P3 stride-8 anchors. The extra 4096 stride-4 anchors add noise without useful signal. **Conclusion: P2 is harmful for cell detection at this image size. Standard 3-scale is optimal.**
- **Run 10 vs Run 7:** Quality focal loss (QFL) caused severe precision collapse (0.695→0.370), even worse than standard focal loss (Run 3: 0.331). mAP@0.5 dropped -0.146. The root cause: YOLO's RayCastAssigner already uses IoU for anchor selection and regression weighting. Using IoU as classification targets creates circular dependency — the classifier is asked to predict what the assigner already selected. **Conclusion: QFL is fundamentally incompatible with IoU-based YOLO assignment. BCE remains the correct choice.**
- **Run 11 vs Run 7:** E2E Fix with Hungarian matching (tal_topk=13, beta=3.0, o2o.topk=1) improved mAP@0.5 by +0.039 (0.389→0.428) and mAP@0.75 by +0.076 (0.174→0.250). DQ remained stable (-0.004) while SQ improved +0.010. The Hungarian matcher provides globally optimal assignment for the one2one branch, enabling NMS-free inference. Lower beta (3.0 vs default 6.0) makes alignment metric less extreme, helping dense touching cells. **Conclusion: E2E Fix with Hungarian matching is a significant improvement, especially at stricter IoU thresholds.**
- **Run 12 vs Run 11:** Large kernel (7×7) in the detection head produced negligible improvement: mAP@0.5 -0.003 (0.428→0.425), SQ +0.003 (0.746→0.749), DQ +0.001 (0.682→0.683). F1 unchanged (0.639). The head-level refinement is NOT the bottleneck — the problem lies further upstream in the feature pyramid / neck. Each detection head already receives features from the PANet neck, so widening the head's receptive field cannot compensate for information lost during neck-level downsampling and fusion. **Conclusion: The bottleneck is in the neck (PANet feature pyramid), not the head. Neck modifications (DWT downsampling, P2 scale, BiFPN fusion) should be prioritized over head modifications.**

---

## 📊 Key Finding: Segmentation vs Detection Quality Gap

### **Run 11 vs LSP-DETR Baseline Comparison**

| Metric | RayCastED (Run 11) | LSP-DETR | Gap | Relative |
|--------|-------------------|----------|-----|----------|
| **SQ** (Segmentation Quality) | **0.746** | 0.811 | -0.065 | **-8.0%** ✅ |
| **DQ** (Detection Quality) | **0.682** | 0.810 | -0.128 | **-15.8%** ❌ |
| mAP@0.5 | 0.428 | 0.691 | -0.263 | -38.1% |
| mAP@0.75 | 0.250 | 0.563 | -0.313 | -55.5% |
| F1 | 0.639 | 0.825 | -0.186 | -22.5% |

### **Thesis Statement**

> **"The segmentation quality is comparable to LSP-DETR but detection quality is severely lacking. Even after changing the matcher to use Hungarian matching for the one2one branch (Run 11), the results don't increase significantly. This might be due to the limitation of relying on YOLO26's original implementation and thus needing further modification in the neck or even the backbone."**

### **Analysis**

1. **SQ Gap (-8.0%)**: RayCastED's polygon shape prediction is competitive
   - SQ = 0.746 (Run 11) vs 0.811 (LSP-DETR)
   - Only 8% gap → geometry prediction is working well
   - Centroid and ray regression are accurate

2. **DQ Gap (-15.8%)**: Detection quality is the bottleneck
   - DQ = 0.682 (Run 11) vs 0.810 (LSP-DETR)
   - 16% gap → assignment/detection is missing cells
   - Problem: Finding cells (DQ) vs drawing accurate polygons (SQ)

3. **Hungarian Matching Impact** (Run 11 vs Run 7):
   - mAP@0.5: +0.039 (9% relative improvement)
   - mAP@0.75: +0.076 (44% relative improvement)
   - DQ: -0.004 (essentially unchanged)
   - **Conclusion:** Hungarian matching helps polygon quality but doesn't fix detection

4. **Root Cause Hypothesis:**
   - YOLO26's feature pyramid (P3-P5) designed for general objects
   - Cell sizes (15-30μm @ 0.25MPP = 60-120px) need specialized scales
   - Current strides (8/16/32) may not capture fine-grained details
   - **Solution direction:** Neck/backbone modifications (P2 scale, large kernels, DWT)

5. **Next Steps:**
   - **Run 12b:** DWT in neck → test if neck downsampling is the bottleneck
   - **Run 13:** P2-P4 scale → finer spatial resolution in the neck
   - **Goal:** Close DQ gap while maintaining SQ advantage

---

## Upcoming Runs

| # | Config | Expected Impact | Evidence | Priority |
|---|--------|----------------|----------|----------|
| ~~12~~ | ~~P3-P5 + Large Kernel (refinement_kernel_size=7)~~ | ~~+0.03-0.06 mAP~~ | ~~RepLKNet +4.2% AP~~ | ✅ DONE: Negligible |
| **12b** | **P3-P5 + Pretrained Backbone (COCO)** | **+0.05-0.15 mAP** | **Transfer learning from COCO** | **HIGH** |
| 13 | P2-P4 scale (yolo26s-p2p4.yaml) | +0.03-0.08 mAP | P2 provides finer resolution | MEDIUM |
| 12c | P2-P4 + Pretrained Backbone | +0.08-0.18 mAP | Combined | MEDIUM |
| 14 | P2-P4 + DWT Neck (DB2 downsampling) | +0.01-0.04 mAP | DWT-UNet +3.2% Dice | MEDIUM |
| 15 | P2-P4 + DCN only (ablation) | +0.03-0.08 mAP | Deformable DETR +8.3% AP | LOW |
| 16 | P2-P4 + DCN + DWT + BiFPN | +0.08-0.15 mAP | BiFPN +2-3% AP (ablation) | LOW |
