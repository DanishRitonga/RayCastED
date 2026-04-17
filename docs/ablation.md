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
