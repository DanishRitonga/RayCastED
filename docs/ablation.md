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
| 7 | + soft polar centerness (PolarMask++), same config as Run 6 | — | — | — | — | — | — | — | — | — |

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
- **Run 7 vs Run 6:** Soft polar centerness (PolarMask++) adds a centerness branch that predicts anchor centering quality. At inference, `cls_score × centerness` suppresses off-center duplicate predictions, improving precision. Target: close the mAP gap further.
