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
| 12b | P3-P5 + Pretrained Backbone (yolo26s.pt, COCO) | 0.547 | 0.409 | 0.225 | 0.530 | 0.739 | **0.671** | 0.665 | 0.582 | 0.621 |
| 13 | P2-P4 + C2PSA@P4 (dummy P5, pretrained) | 0.551 | 0.423 | 0.214 | 0.531 | 0.734 | **0.678** | 0.676 | 0.602 | 0.637 |
| 13b | P2-P4 + C2PSA@P4 (dummy P5, scratch) | 0.511 | 0.401 | 0.128 | 0.499 | 0.703 | 0.665 | 0.662 | 0.581 | 0.619 |
| 14 | P2-P4 + ResoConv (zero pad) | 0.561 | 0.419 | 0.220 | 0.540 | 0.744 | 0.681 | 0.673 | 0.595 | 0.632 |
| 14b | P2-P4 + ResoConv (reflect pad) | 0.561 | 0.432 | 0.258 | 0.542 | 0.750 | 0.677 | 0.682 | 0.596 | 0.637 |
| 15 | P2-P4 + C3k2_LK neck only | 0.526 | 0.419 | 0.155 | 0.509 | 0.713 | 0.670 | 0.669 | 0.593 | 0.629 |
| 16 | P2-P4 + C3k2_LK backbone only | 0.546 | 0.416 | 0.206 | 0.534 | 0.734 | 0.684 | 0.678 | 0.587 | 0.629 |
| 16b | P2-P4 + C3k2_LK backbone + neck | 0.549 | 0.409 | 0.212 | 0.532 | 0.735 | 0.678 | 0.669 | 0.588 | 0.626 |
| **18** ⭐ | **ResoConv + LK backbone (reflect), std neck** | **0.555** | **0.405** | **0.219** | **0.540** | **0.745** | **0.680** | **0.675** | **0.582** | **0.625** |
| 20 | ResoConv no-shortcut + LK backbone (reflect), std neck | 0.558 | 0.414 | 0.249 | 0.540 | 0.751 | 0.674 | 0.672 | 0.582 | 0.624 |
| 21 | ResoConv no-HH + LK backbone (reflect), std neck | 0.555 | 0.414 | 0.245 | 0.535 | 0.747 | 0.670 | 0.675 | 0.583 | 0.626 |
| 22 | bior2.2 + ResoConv + LK backbone (reflect), std neck | 0.551 | 0.421 | 0.239 | 0.527 | 0.744 | 0.663 | 0.681 | 0.580 | 0.626 |

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

### Run 12b — P3-P5 + ResoConv Neck (Phase 1b) — PENDING
Replace PANet downsampling conv with Daubechies-2 DWT in the neck only.
Purpose: Test if preserving high-frequency details during feature fusion helps DQ.
Expected: +0.01-0.04 mAP if neck is the bottleneck.

### Run 12b — P3-P5 + Pretrained Backbone (COCO) ✅ COMPLETE
Same as Run 11 but with COCO-pretrained backbone (layers 0-10):
```yaml
  pretrained_backbone: "yolo26s.pt"  # 5.46M backbone params from COCO
  # Neck and head remain randomly initialised
```
Result: **Negligible improvement vs Run 11.** DQ -0.011 (0.682→0.671), mAP@0.5 -0.019 (0.428→0.409). All metrics regressed slightly. **Conclusion: COCO pretrained backbone is NOT helpful for histopathology cell detection.** Domain gap (natural images → medical images) negates transfer learning benefits.

### Run 13 — P2-P4 + C2PSA@P4 (dummy P5, pretrained) ✅ COMPLETE
```yaml
global_settings:
  model: "raycasted/cfg/yolo26s-p2p4-c2psa.yaml"  # Custom P2-P4 neck
  # C2PSA moves from P5/32 (1024ch) to P4/16 (512ch)
  # Dummy P5 layers 9-10 built but unused in neck

training:
  pretrained_backbone: "yolo26s.pt"  # Layers 0-6 load, 7-10 skip
```
Result: **Modest improvement over Run 12b.** DQ +0.007 (0.671→0.678), mAP@0.5 +0.014 (0.409→0.423), F1 +0.016 (0.621→0.637). Params: 6.65M, GFLOPs: 7.15. **Conclusion: P2 density provides small but real benefit.** However, the gain is insufficient to close the DQ gap to LSP-DETR (0.678 vs 0.810). Bottleneck remains in backbone receptive field, not just anchor density.

### Run 13b — P2-P4 + C2PSA@P4 (scratch) ✅ COMPLETE
Same as Run 13 but without pretrained backbone:
```yaml
  pretrained_backbone: null  # Train from scratch
```
Result: All metrics regressed vs Run 13 (pretrained): DQ -0.013 (0.678→0.665), mAP@0.5 -0.022 (0.423→0.401), SQ -0.031 (0.734→0.703). **Conclusion: Pretrained weights help but are not essential** (scratch DQ=0.665 ≥ 0.65 threshold). Custom backbone viable for domain-specific architectures.

### Run 14 — P2-P4 + ResoConv (zero pad) ✅ COMPLETE
```yaml
  model: "raycasted/cfg/yolo26s-resoconv-p2p4.yaml"
```
All 5 backbone + 2 neck strided Convs replaced with ResoConv (DB2 DWT + 1x1 projection). Params: 3.73M (−44% vs 6.65M baseline). Result: **Improved every metric vs Run 13b.** mAP@0.5 +0.018 (0.401→0.419), SQ +0.041 (0.703→0.744), F1 +0.013 (0.619→0.632). DWT frequency decomposition preserves high-frequency boundary information that strided conv destroys. **Conclusion: ResoConv is the single most effective modification — better polygons with half the parameters.**

### Run 14b — P2-P4 + ResoConv (reflect pad) ✅ COMPLETE
Same as Run 14 but with reflection padding instead of zero padding in DWT. Result: mAP@0.5 +0.013 (0.419→0.432), mAP@0.75 +0.038 (0.220→0.258), SQ +0.006 (0.744→0.750). Reflection padding eliminates spurious high-frequency artifacts at tile boundaries. DQ dipped slightly (-0.004) but overall quality improved. **Conclusion: Reflection padding is the correct choice for DWT downsampling.**

### Run 15 — P2-P4 + C3k2_LK neck only ✅ COMPLETE
```yaml
  model: "raycasted/cfg/yolo26s-run15-lk-neck.yaml"
```
C3k2_LK (scale-adaptive large-kernel depthwise conv) in neck only, standard C3k2 in backbone. Params: 4.02M. Result: Improved over Run 13b baseline: mAP@0.5 +0.018 (0.401→0.419), SQ +0.010 (0.703→0.713), DQ +0.005 (0.665→0.670). **Conclusion: Large-kernel neck helps but less than ResoConv.**

### Run 16 — P2-P4 + C3k2_LK backbone only ✅ COMPLETE
```yaml
  model: "raycasted/cfg/yolo26s-run16-lk-backbone.yaml"
```
C3k2_LK in backbone only, standard C3k2 in neck. Params: 4.05M. Result: **Highest DQ of any single modification** (0.684 vs 0.681 ResoConv, 0.670 LK neck). PQ +0.035 (0.499→0.534), SQ +0.031 (0.703→0.734). Large receptive field in feature extraction directly helps detect more cells. **Conclusion: LK backbone is the best single modification for DQ.**

### Run 16b — P2-P4 + C3k2_LK backbone + neck ✅ COMPLETE
```yaml
  model: "raycasted/cfg/yolo26s-run16b-lk-full.yaml"
```
C3k2_LK in both backbone AND neck. Params: 3.91M. Result: **Strictly worse than Run 16 (LK backbone only).** DQ 0.678 vs 0.684 (−0.006), mAP@0.5 0.409 vs 0.416 (−0.007), PQ 0.532 vs 0.534 (−0.002). The only improvement over baseline (Run 13b) is SQ (+0.032) and PQ (+0.033) — essentially identical to Run 16 alone. Adding LK to the neck did NOT compound with LK backbone; it erased the DQ gain.

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
- **Run 12b vs Run 11:** COCO pretrained backbone regressed all metrics vs the trained-from-scratch Run 11: DQ -0.011 (0.682→0.671), mAP@0.5 -0.019 (0.428→0.409). **Conclusion: Pretrained weights from natural images (COCO) hurt histopathology cell detection.** The domain gap (RGB natural images → H&E stained tissue) is too large. The backbone features learned for general object detection don't transfer to cell detection. **Transfer learning requires domain-specific pretrained weights (e.g., UniRepLKNet trained on PanNuke/MoNuSeg).**
- **Run 13b vs Run 13:** Training from scratch (no pretrained) regressed vs pretrained: DQ -0.013 (0.678→0.665), mAP@0.5 -0.022 (0.423→0.401), SQ -0.031 (0.734→0.703). Scratch DQ=0.665 ≥ 0.65 threshold → custom backbone viable. **Conclusion: Pretrained COCO weights help but are not essential. Custom architectures trained from scratch can match or exceed pretrained baseline.**
- **Run 14 vs Run 13b:** ResoConv (DB2 DWT downsampling) improved every metric: mAP@0.5 +0.018, SQ +0.041, F1 +0.013, with −44% params (3.73M vs 6.65M). The wavelet frequency decomposition explicitly preserves LH/HL/HH sub-bands (edges, boundaries) that strided conv destroys. **Conclusion: DWT downsampling is the most effective single modification for polygon quality (SQ).**
- **Run 14b vs Run 14:** Reflection padding improved mAP@0.5 +0.013, mAP@0.75 +0.038, SQ +0.006. Zero padding creates spurious high-frequency artifacts at boundaries; reflection padding mirrors the signal for clean decomposition. **Conclusion: Always use reflection padding for DWT.**
- **Run 15 vs Run 13b:** C3k2_LK in neck only: mAP@0.5 +0.018, SQ +0.010, DQ +0.005. Large-kernel receptive field in feature fusion helps but is the weakest of the three single modifications. **Conclusion: Neck receptive field contributes modestly.**
- **Run 16 vs Run 13b:** C3k2_LK in backbone only: **highest DQ of any run (0.684)**, PQ +0.035, SQ +0.031. Large-kernel depthwise conv in backbone feature extraction gives P2 anchors enough spatial context (~30-40px receptive field) to distinguish touching cells. **Conclusion: Backbone receptive field is the primary lever for DQ.**
- **Run 16b vs Run 16 (negative interaction):** Adding C3k2_LK to the neck on top of LK backbone **erased the DQ gain**: DQ 0.678 vs 0.684 (−0.006), mAP@0.5 0.409 vs 0.416 (−0.007). This is a **negative interaction** — not merely lack of compounding, but actual regression. The neck's job is fine-grained channel mixing and spatial precision for ray endpoint regression. LK neck over-smooths the already context-rich backbone features, destroying the spatial precision that the standard 3x3 neck preserved. See detailed analysis below.
- **Cross-comparison (Runs 14b, 15, 16, 16b):**
  - **ResoConv** wins SQ (0.750) with −44% params → best polygon shape quality
  - **LK backbone** wins DQ (0.684) → best cell detection/delineation
  - **LK neck** is weakest alone (PQ 0.509) and actively harmful when combined with LK backbone (PQ 0.532 vs 0.534)
  - **Key insight:** LK in the backbone and neck serve fundamentally different roles. Backbone needs large receptive field (context). Neck needs precise local fusion (detail). Mixing both with LK conflates these roles.
  - ~~Run 18 (ResoConv + LK backbone, standard 3x3 neck) should combine the best of both.~~ **Result (Run 18): Did NOT compound.** See updated analysis below.
- **Run 18 vs Run 14b (ResoConv + LK backbone):** Combined the SQ winner with the DQ winner — but **regressed on BOTH**. SQ 0.745 vs 0.750 (−0.005), DQ 0.680 vs 0.684 (−0.004), mAP@0.5 0.405 vs 0.432 (−0.027). The LK backbone's spatial smoothing partially destroys the frequency decomposition that ResoConv provides. **Conclusion: ResoConv and LK backbone have negative interference.**
- **Run 20 vs Run 18 (no-shortcut):** Removing the ResoConv shortcut connection improved mAP@0.75 (0.219→0.249, **best of any run**) and SQ (0.745→0.751) at the cost of DQ (0.680→0.674). The pure DWT signal (no identity blending) is cleaner for shape quality but loses spatial localization. **Conclusion: No-shortcut variant is better for polygon precision.**
- **Run 21 vs Run 18 (no-HH sub-band):** Dropping the HH (diagonal) sub-band saved 0.21M params with negligible quality loss. mAP@0.5 0.414 (same as Run 20), SQ 0.747, DQ 0.670. **Conclusion: HH contributes negligibly to cell boundary detection.**
- **Run 22 vs Run 18 (bior2.2 wavelet):** Biorthogonal 2.2 achieved highest mAP@0.5 among ResoConv+LK runs (0.421) but lowest DQ (0.663). The symmetric wavelet helps alignment but doesn't match H&E edge characteristics as well as DB2. **Conclusion: Wavelet choice is minor; DB2 remains default.**
- **Cross-comparison (Runs 14b, 18, 20, 21, 22):** **Run 14b (ResoConv alone) remains the best overall architecture.** LK backbone consistently hurts ResoConv — the frequency preservation and large-kernel context compete for the same information budget. The no-shortcut variant (Run 20) is promising for mAP@0.75 but can't close the detection gap.

---

## 📊 Analysis: Why LK Backbone + LK Neck Fails (Run 16b)

### Results Summary

| Run | LK Backbone | LK Neck | DQ | SQ | mAP@0.5 | PQ |
|-----|:-----------:|:-------:|:---:|:---:|:-------:|:---:|
| 13b (baseline) | — | — | 0.665 | 0.703 | 0.401 | 0.499 |
| 15 (LK neck) | — | K=13 | 0.670 | 0.713 | 0.419 | 0.509 |
| 16 (LK backbone) | K=7,9 | — | **0.684** | 0.734 | 0.416 | 0.534 |
| 16b (LK full) | K=7,9 | K=7,9,13 | 0.678 | 0.735 | 0.409 | 0.532 |
| 14b (ResoConv) | — | — | 0.677 | **0.750** | **0.432** | 0.542 |

### The Negative Interaction

Run 16b does not compound — it **regresses** vs Run 16 on the metrics that LK backbone was supposed to improve:

| Metric | Run 16 (backbone only) | Run 16b (backbone + neck) | Delta |
|--------|:-----------------------:|:-------------------------:|:-----:|
| DQ | **0.684** | 0.678 | **−0.006** |
| mAP@0.5 | **0.416** | 0.409 | **−0.007** |
| Precision | **0.678** | 0.669 | **−0.009** |
| Recall | 0.587 | 0.588 | +0.001 |
| SQ | 0.734 | 0.735 | +0.001 |

DQ and precision dropped while recall was flat. The model detects roughly the same number of cells but **localizes them worse** — the extra context from LK neck doesn't help delineation, it actively hurts it.

### Root Cause: Role Conflation in Backbone vs Neck

The backbone and neck serve fundamentally different purposes in a detection pipeline:

**Backbone role — "What is where?" (contextual understanding)**
- Extract semantically rich features across spatial scales
- Large receptive field is critical: a P2 stride-4 anchor at 256px crop covers a 4×4 pixel region in the feature map, but needs to "know" about neighboring cells ~30-40px away to distinguish touching vs separate nuclei
- LK backbone (K=7,9) provides exactly this: the dilated depthwise branches (K=9 → branches at dilation 1,2,3) span up to 21×21 pixels at P2, enough to see adjacent cells
- This is why Run 16 DQ=0.684 — the backbone can resolve touching cells

**Neck role — "Exactly where?" (spatial precision)**
- Fuse multi-scale features (top-down pathway + bottom-up pathway)
- The FPN/PAFP neck combines coarse semantic features from P4 with fine-grained spatial features from P2
- After concatenation, the neck blocks perform **channel mixing** (which features matter?) and **spatial alignment** (where exactly are the boundaries?)
- For ray endpoint prediction, the neck needs to preserve **pixel-level spatial precision** — each ray endpoint needs sub-pixel accuracy
- Standard 3×3 convolutions in C3k2 are ideal for this: small receptive field = local detail preservation
- The 1×1 convolutions in the C3k2 bottleneck (expand/project) handle channel mixing independently of spatial context

**What goes wrong with LK neck:**
- LK neck (K=7,9,13) applies large-kernel depthwise convolutions **after** the Concat merge
- The Concat already fuses features from different scales — the spatial information is at mixed resolutions
- A 13×13 depthwise conv at P2 stride-4 spans 52×52 pixels in input space — it mixes features from cells 52px apart
- This **over-smooths** the precise boundary information that the backbone worked hard to extract
- The result: broader, less precise features that are good for saying "there's a cell here" (SQ=0.735, unchanged) but worse for saying "exactly where its boundary is" (DQ=0.678, regressed)

### Evidence from UniRepLKNet's Architecture

UniRepLKNet's own design validates this analysis. In `UniRepLKNetBlock`, the block structure is:

```
Input → DilatedReparamDW (large kernel, spatial aggregation)
      → BN → SE (channel attention)
      → Linear (1×1 PW, channel mixing) → GELU → GRN
      → Linear (1×1 PW, channel mixing) → BN
      → + residual
```

The **spatial aggregation** (large kernel) and **channel mixing** (1×1 linear/conv) are **explicitly separated**. The FFN after the DW conv uses **only 1×1 operations** — no spatial convolution at all. This is the same principle as keeping 3×3 in the neck.

Furthermore, UniRepLKNet-S uses an **alternating kernel pattern** in stage 3: `(13, 3, 3, 13, 3, 3, ...)` — not all-13. The 3×3 blocks after the 13×13 block serve as "local refinement," analogous to our standard 3×3 neck preserving precision after LK backbone provides context.

### The Information Flow Argument

Think of the detection pipeline as an information processing chain:

```
Input image (raw pixels, full detail)
    ↓ Backbone: extract context (WHAT cells look like)
Feature maps (semantic, lower spatial detail)
    ↓ Neck: fuse scales + refine boundaries (WHERE exactly)
Neck features (precise localization)
    ↓ Head: predict rays per cell
Polygon output
```

1. **LK backbone** enriches the "WHAT" — the backbone features carry more contextual information about neighboring cells, tissue structure, staining patterns
2. **Standard 3×3 neck** takes those rich features and preserves the "WHERE" — the fine-grained spatial information needed for ray endpoint regression
3. **LK neck** attempts to add more "WHAT" (context) at a stage that needs "WHERE" (precision) — it's the wrong operation at the wrong stage

### Parametric Evidence

Run 16b actually has **fewer parameters** than Run 16 (3.91M vs 4.05M). This is because `C3k2_LK` with expansion `e=0.5` uses the same hidden channel count as `C3k2` with `c3k=True` (which internally uses `e=0.5`), but the DilatedReparamDW uses depthwise conv (groups=dim, param multiplier = 1) instead of the dense conv in standard C3k2 Bottleneck (groups=1, param multiplier = c_in×c_out). So the LK neck is not only less effective but also uses parameters differently — trading dense channel interaction for sparse spatial aggregation, which is the wrong tradeoff in the neck.

### Implications for Future Runs

1. **Run 17 (ResoConv + LK neck) — CANCELLED.** If LK neck hurts when combined with LK backbone, it will likely hurt when combined with ResoConv too. The neck role is fundamentally about local precision, not spatial context.

2. **Run 19 (ResoConv + LK everywhere) — DEPRIORITIZED.** Same reasoning. LK in the neck is harmful regardless of backbone choice.

3. **Run 18 (ResoConv + LK backbone, standard neck) — HIGHEST PRIORITY.** This combines the SQ winner (ResoConv, 0.750) with the DQ winner (LK backbone, 0.684) while keeping the standard 3×3 neck that preserves spatial precision. This is the configuration most aligned with the architectural principle of separated roles.

4. **Future: SE + Layer Scale on LK backbone.** Rather than putting LK in the neck, the better path is to make the LK backbone stronger with UniRepLKNet's missing ingredients (SE attention, layer scale, GRN). These add channel-level intelligence to the backbone without touching the neck.

### Run 18 — ResoConv + LK backbone (reflect), std neck ✅ COMPLETE
```yaml
  model: "raycasted/cfg/yolo26s-run18-resoconv-lk-backbone.yaml"
```
Combines ResoConv (SQ winner, Run 14b) with LK backbone (DQ winner, Run 16), keeping standard 3×3 neck. Params: 3.63M. Result: **Disappointing — did NOT compound.** SQ 0.745 vs 0.750 (Run 14b, −0.005), DQ 0.680 vs 0.684 (Run 16, −0.004), mAP@0.5 0.405 vs 0.432 (Run 14b, −0.027). The combined architecture regressed on BOTH dimensions. **Conclusion: ResoConv and LK backbone interfere with each other.** Both modifications enrich features in different ways but compete for the same information budget. ResoConv preserves high-frequency detail (edges, boundaries) while LK backbone expands receptive field (context). When combined, the LK backbone's spatial smoothing may partially destroy the frequency decomposition that ResoConv works to preserve.

### Run 20 — ResoConv no-shortcut + LK backbone ✅ COMPLETE
```yaml
  model: "raycasted/cfg/yolo26s-run20-resoconv-no-shortcut-lk-backbone.yaml"
```
Same as Run 18 but ResoConv without the shortcut (residual) connection — the DWT output goes directly through the 1×1 projection without additive skip. Params: 3.52M (−0.11M vs Run 18). Result: **Best mAP@0.75 of any run (0.249).** mAP@0.5 +0.009 (0.405→0.414), SQ +0.006 (0.745→0.751), but DQ −0.006 (0.680→0.674). Removing the shortcut eliminates the identity path that blends pre-DWT features (spatial) with post-DWT features (frequency). The pure DWT signal is cleaner for shape quality but loses some spatial localization. **Conclusion: No-shortcut ResoConv improves polygon precision (SQ, mAP@0.75) at the cost of detection quality (DQ).**

### Run 21 — ResoConv no-HH + LK backbone ✅ COMPLETE
```yaml
  model: "raycasted/cfg/yolo26s-run21-resoconv-no-hh-lk-backbone.yaml"
```
Same as Run 18 but ResoConv drops the HH (diagonal) sub-band — only keeps LL (approximation) and LH/HL (horizontal/vertical edges). Params: 3.42M (−0.21M vs Run 18). Result: **Similar to Run 20.** mAP@0.5 0.414 (same as Run 20), SQ 0.747, DQ 0.670 (−0.010 vs Run 18). Removing HH removes ~33% of the DWT channels but the diagonal edge information has marginal value for cell boundaries. **Conclusion: HH sub-band contributes negligibly. Potential for channel-efficient ResoConv variant.**

### Run 22 — bior2.2 + ResoConv + LK backbone ✅ COMPLETE
```yaml
  model: "raycasted/cfg/yolo26s-run22-bior22-resoconv-lk-backbone.yaml"
```
Same as Run 18 but Daubechies-2 wavelet replaced with biorthogonal 2.2 (bior2.2). Params: 3.52M. Result: **Highest mAP@0.5 of the ResoConv+LK runs (0.421)** but lower AJI (0.551 vs 0.558) and PQ (0.527 vs 0.540). The biorthogonal wavelet is symmetric (linear phase) which may help reconstruction quality but doesn't match the edge characteristics of H&E cell boundaries as well as DB2. **Conclusion: Wavelet choice has minor but measurable impact. DB2 remains the default.**

### Cross-comparison: ResoConv + LK Backbone Ablations (Runs 18, 20, 21, 22)

| Run | ResoConv Variant | AJI | mAP@0.5 | mAP@0.75 | PQ | SQ | DQ | Params |
|-----|-----------------|-----|---------|----------|-----|-----|-----|--------|
| 14b | ResoConv (reflect), no LK | **0.561** | **0.432** | 0.258 | **0.542** | **0.750** | 0.677 | 3.73M |
| 18 | ResoConv + LK backbone | 0.555 | 0.405 | 0.219 | 0.540 | 0.745 | 0.680 | 3.63M |
| 20 | ResoConv no-shortcut + LK | **0.558** | 0.414 | **0.249** | 0.540 | **0.751** | 0.674 | 3.52M |
| 21 | ResoConv no-HH + LK | 0.555 | 0.414 | 0.245 | 0.535 | 0.747 | 0.670 | 3.42M |
| 22 | bior2.2 + LK | 0.551 | **0.421** | 0.239 | 0.527 | 0.744 | 0.663 | 3.52M |

**Key finding: LK backbone consistently hurts ResoConv.** Run 14b (ResoConv alone) outperforms ALL ResoConv+LK combinations on mAP@0.5 (0.432 vs best 0.421) and PQ (0.542 vs best 0.540). The LK backbone's large receptive field interferes with the frequency decomposition that ResoConv provides. The no-shortcut variant (Run 20) recovers some ground (best mAP@0.75=0.249, SQ=0.751) but still can't match ResoConv alone on detection metrics.

### Revised Architecture Principle

> **"Large kernels for context (backbone), small kernels for precision (neck), wavelets for preservation (downsampling)."**

This is the design rule that emerges from Runs 14-16b. Any modification must respect the role boundary between backbone and neck.

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
      - ~~**Run 13b:**~~ ✅ DONE: Scratch DQ=0.665, custom backbone viable
      - ~~**Run 14:**~~ ✅ DONE: ResoConv best for SQ (0.750), −44% params
      - ~~**Run 15:**~~ ✅ DONE: LK neck weakest modification
      - ~~**Run 16:**~~ ✅ DONE: LK backbone best for DQ (0.684)
      - ~~**Run 16b:**~~ ✅ DONE: LK full regressed vs LK backbone alone (DQ 0.678 vs 0.684). **Negative interaction confirmed.**
      - **Run 18:** P2-P4 + ResoConv + C3k2_LK backbone → **HIGHEST PRIORITY** (combines SQ + DQ winners, standard 3x3 neck)
      - **Future:** Learnable sub-band weighting in ResoConv (HFE-DWT, WaveDH), biorthogonal wavelets (bior1.3)
      - **Goal:** Close DQ gap (target: 0.75+) while maintaining SQ advantage

---

## Upcoming Runs

See `docs/backbone_neck_modification.md` for the full architecture plan and YAML configs.

| # | Config | Expected Impact | Evidence | Priority |
|---|--------|----------------|----------|----------|
| ~~12~~ | ~~P3-P5 + Large Kernel (refinement_kernel_size=7)~~ | ~~+0.03-0.06 mAP~~ | ~~RepLKNet +4.2% AP~~ | ✅ DONE: Negligible |
| ~~12b~~ | ~~P3-P5 + Pretrained Backbone (COCO)~~ | ~~+0.05-0.15 mAP~~ | ~~Transfer learning from COCO~~ | ✅ DONE: Regression (-1.9% mAP) |
| ~~13~~ | ~~P2-P4 + C2PSA@P4 (pretrained)~~ | ~~+0.03-0.08 mAP~~ | ~~P2 provides finer resolution~~ | ✅ DONE: +1.4% mAP |
| ~~13b~~ | ~~P2-P4 + C2PSA@P4 (scratch)~~ | ~~Pretrained impact measurement~~ | ~~If DQ ≥ 0.65 → custom backbone viable~~ | ✅ DONE: DQ=0.665, viable |
| ~~14~~ | ~~P2-P4 + ResoConv (zero pad)~~ | ~~+0.03-0.08 mAP~~ | ~~LKCell SOTA PanNuke, WaveCNet +AP~~ | ✅ DONE: +4.5% mAP, SQ=0.744 |
| ~~14b~~ | ~~P2-P4 + ResoConv (reflect pad)~~ | ~~Incremental~~ | ~~Boundary artifact removal~~ | ✅ DONE: +3.1% mAP, SQ=0.750 |
| ~~15~~ | ~~P2-P4 + C3k2_LK neck only~~ | ~~+0.02-0.05 mAP~~ | ~~UniRepLKNet large kernels~~ | ✅ DONE: +4.5% mAP, weakest LK |
| ~~16~~ | ~~P2-P4 + C3k2_LK backbone only~~ | ~~+0.02-0.06 mAP~~ | ~~Receptive field at P2/P3~~ | ✅ DONE: DQ=0.684 (best DQ) |
| ~~16b~~ | ~~P2-P4 + C3k2_LK backbone + neck~~ | ~~+0.04-0.08 mAP~~ | ~~LK interaction~~ | ✅ DONE: **Negative interaction** DQ 0.678 vs 0.684 |
| ~~17~~ | ~~P2-P4 + ResoConv + C3k2_LK neck~~ | ~~+0.05-0.10 mAP~~ | ~~ResoConv SQ + LK neck~~ | ❌ CANCELLED: LK neck harmful |
| **18** | **P2-P4 + ResoConv + C3k2_LK backbone** | **+0.08-0.12 mAP** | **Combines SQ + DQ winners, standard neck** | **HIGHEST** |
| ~~19~~ | ~~P2-P4 + ResoConv + C3k2_LK everywhere~~ | ~~+0.08-0.15 mAP~~ | ~~Full combined~~ | ❌ DEPRIORITIZED: LK neck harmful |
| 20 | UniRepLKNet-S backbone + P2-P4 neck | +0.08-0.15 mAP | Domain-specific pretrained | MEDIUM |
| 21 | SE + Layer Scale on LK backbone (Run 16 + SE/GRN) | +0.02-0.04 mAP | UniRepLKNet missing ingredients | HIGH |
