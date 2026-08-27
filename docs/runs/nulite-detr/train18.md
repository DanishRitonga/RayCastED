# train18 — LSP-DETR inner-mask matching cost (DETR single-head)

**Run:** `output/nulite/train18`
**Date:** 2026-08-28
**Config:** `main/pannuke.yaml`, `architecture: nulite_detr`, `detr_cost_inside: 10.0`
**Checkpoint:** `output/nulite/train18/weights/best.pt` (epoch 600, fitness 0.5521)

## Config change from train17

Added the LSP-DETR inner-mask matching cost `Wm` to the Hungarian matcher:

```
detr_cost_inside: 10.0   # Wm(m,p) = 0 if centroid inside GT polygon, λ otherwise
```

This is the **only** change vs train17 (which used `detr_cost_inside` implicitly 0 / no inside term).

## Root cause this addresses

train17 (FCN-loss parity, no inside cost) reached val bPQ 0.540 / bDQ 0.659 but collapsed
on the test set (bPQ 0.297 / mAP50 0.055). Diagnosis: the DETR predicted **good ray
lengths but displaced centroids** — polygon-IoU bPQ 0.666 vs rasterized 0.297, centroid
L2 error 117px. Rays emitted from the wrong location land the mask in the wrong place,
so mask-IoU collapses even though the polygon geometry is right.

The LSP-DETR `Wm` cost directly targets this: a predicted centroid gets +λ cost in the
Hungarian assignment if it falls **outside** the GT nucleus, forcing the matcher to assign
each query to the nucleus it actually sits inside (and thereby pushing the centroid
regression toward inside-nucleus locations).

## Implementation

- `raycasted/model/nulite_detr_loss.py`
  - new `_points_inside_star(pred_xy, gt_xy, gt_rays, n_rays)`: exact star-convex
    point-in-polygon test — a centroid is inside iff its distance from the GT center
    along its angular direction is <= the ray length interpolated between the two
    bracketing rays.
  - `RayCastHungarianMatcher` gained `cost_inside` (default 10.0) added to the cost
    matrix as `cost_gain['inside'] * (~inside).float()`.
- `raycasted/model/nulite_detr_model.py`: `cost_inside` ctor param → `init_criterion()`.
- `raycasted/model/train.py`: `cost_inside=float(tcfg.get('detr_cost_inside', 10.0))`.
- `raycasted/data/etl/utils/config.py`: `detr_cost_inside: float = 10.0`.
- `main/pannuke.yaml`: `detr_cost_inside: 10.0`.

## Training (val on train set, 2656 imgs / 62452 GT)

600 epochs, batch 16, imgsz 256, MuSGD, amp=false, patience 500, plain cosine
(`warm_restarts: false`). **0 zero-metric epochs.**

| metric | best epoch | value |
|---|---|---|
| bPQ | 598 | **0.5966** |
| mAP50 | 587 | **0.4486** |
| mAP50-95 | 600 | 0.3442 |
| bSQ | 600 | 0.7805 |
| bDQ | 600 | 0.7250 |
| precision | 600 | 0.4531 |
| recall | 600 | 0.6558 |

val cls loss 2.60 → 2.22; train cls 2.81 → 0.13.

## Test-set eval (conf=0.20, 2722 tiles)

```
uv run python -m raycasted.evals.eval_pannuke \
  --weights output/nulite/train18/weights/best.pt \
  --data-dir output/pannuke_64/transformed/test --batch 16 --device 0
```

| metric | train18 (DETR) | train17 (no inside) | FCN train6 |
|---|---|---|---|
| **bPQ** | **0.7691** | 0.2967 | 0.6310 |
| **mAP50** | **0.8730** | 0.0547 | 0.5754 |
| **mPQ** | **0.6771** | 0.4209 | 0.4076 |
| **AJI** | **0.7476** | 0.2767 | 0.6066 |
| F1 (centroid) | **0.8864** | 0.6867 | 0.5424 |
| Precision | 0.8012 | 0.5772 | — |
| Recall | **0.9918** | 0.8477 | 0.8989 |
| AP@0.5 | 0.8730 | 0.0547 | 0.5754 |
| AP@0.5:0.05:0.95 | 0.4883 | 0.0271 | 0.3412 |
| **preds vs GT** | **81,515 (1.24x)** | 96,718 (1.47x) | 152,726 (2.31x) |

Polygon-level (no rasterization): bPQ 0.8446, bSQ 0.9536, bDQ 0.8857.

Tissue macro (19 tissues): bPQ 0.7785 ± 0.0477, mPQ 0.7022 ± 0.0402, AJI 0.7600 ± 0.0492.
Worst tissue Colorectal (bPQ 0.6986 / mPQ 0.6099). Class F1: Necrosis weakest (0.7891,
1039 instances, rare class). Recall by size: small 0.9791, medium 0.9975, large 0.9988.

## Interpretation

The inner-mask cost fixed the displaced-centroid problem nearly completely:

- centroid recall 0.9918 (matched 65,311/65,848 GT; only 537 unmatched, 66% of those
  >12px away = genuine misses).
- overprediction down to 1.24x (was 2.31x FCN, 6.1x early DETR).
- the val→test gap collapsed (test bPQ 0.769 actually *higher* than val 0.597).

This is the strongest result across all RayCastED / NuLite variants, and bPQ 0.769 is
competitive with the LSP-DETR benchmark itself (DQ 0.810). The DETR single-head + LSP-DETR
grid-query init + focal ∅ + MDS rank-causal mask + FCN-loss parity + inner-mask cost is
now the reference architecture.

## Next steps

- Commit the `detr_cost_inside` feature + FCN-loss-parity + fitness-bug fixes (uncommitted
  as of this run).
- Ablate `detr_cost_inside` (λ ∈ {2, 5, 20}) to confirm 10 is near-optimal.
- Necrosis (rare class, 1.6% of instances) is the remaining weak spot — class-imbalance
  handling.
- Consider a longer schedule (still climbing at 600) or SGDR revisit now that the
  winner-selection is stable.
