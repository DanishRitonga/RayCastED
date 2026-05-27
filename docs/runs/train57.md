# Train57 — Hungarian AND Gate + Soft Penetration

**Date**: 2026-05-27
**Config diff from Train49**: Hungarian enabled (phase2_start=300, cost_inner=9999, cost_ray_quality=1.0, cost_inner_sigma=0.1, cls_only=True), bg_fg=0

## Results (early stopped ~308, catastrophic collapse after)

| Metric | Value (epoch 308) |
|--------|-------------------|
| mAP50 | 0.525 |
| mAP50-95 | 0.413 |
| P | 0.633 |
| R | 0.414 |
| Cent mAP50 | 0.508 |

## Hungarian DIAG at epoch 308

- hw=0.063 (only 6.3% weight)
- Raw losses: cls=119-142, L1=52-59, piou=94-105
- TAL raw losses for comparison: cls=0.75, piou=0.25

## Diagnosis

Model peaked at epoch 308 (barely after Hungarian started ramping at 300). As hw increased toward 0.9, catastrophic collapse to ~0 mAP. The AND gate (outside+wrong class → +9999, outside+correct → small) fixed spatial validity but Hungarian still produces terrible cls targets (142x TAL's cls loss). Even cls_only blending couldn't prevent destruction.

o2o fg/bg gap at peak: fg=0.741, bg=0.008 → 92x (absurdly distorted from train49's 2.5x).

**Conclusion**: Hungarian is a dead end for FCN. No cost function tweak can fix it — the problem is architectural. Hungarian matches anchors with terrible regression (piou=94) and tells cls head "be confident." In FCN, bad cls targets propagate everywhere. 4th failed Hungarian experiment (train50, train53, train57, plus cls-only variant).
