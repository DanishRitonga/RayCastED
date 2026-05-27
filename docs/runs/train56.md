# Train56 — bg_fg_ratio_o2o=1.0 Curriculum

**Date**: 2026-05-27
**Config diff from Train49**: bg_fg_ratio_o2o=1.0 (curriculum target), bg_fg_ratio_o2o_curriculum_epoch=220, Hungarian disabled

## Results (400 epochs, best=400)

| Metric | Value |
|--------|-------|
| mAP50 | 0.517 |
| mAP50-95 | 0.400 |
| P | 0.519 |
| R | 0.510 |
| Cent mAP50 | 0.501 |
| Cent R | 0.534 |

## Comparison

| Metric | Train49 (bg_fg=0) | Train56 (curr target 1.0) | Delta |
|--------|-------------------|--------------------------|-------|
| mAP50 | 0.530 | 0.517 | -0.013 |
| mAP50-95 | 0.403 | 0.400 | -0.003 |
| R | 0.533 | 0.510 | -0.023 |
| Cent R | 0.533 | 0.534 | +0.001 |

## Diagnosis

bg_fg curriculum target=1.0 **hurt** vs train49's bg_fg=0. fg/bg gap collapsed from 2.5x to 1.43x as curriculum ramped bg_fg after epoch 220. Centroid recall slightly improved (+0.001) — bg gradient helps localization — but classification suffered.

**Conclusion**: Confirms bg_fg > 0 in o2o always hurts fg/bg discrimination. Train49's bg_fg=0 remains optimal.
