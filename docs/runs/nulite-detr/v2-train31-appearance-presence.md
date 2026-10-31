# V2 Train31 — supervised query appearance + presence

- Output: `output/v2/train31`
- Checkpoint evaluated: `weights/last.pt`
- Stop epoch: **332** of the unchanged 600-epoch cosine schedule.

## Change from Train30

Train31 retains the supervised stride-2 appearance map and polygon-conditioned
query injection from Train30. It adds a training-only balanced query-presence
BCE with weight 0.1 after the appearance residual. Queries are positive when
their predicted center lies inside a GT and that GT center lies inside their
predicted polygon. Queries at least four pixels outside all GTs and containing
no GT centroid are negatives; ambiguous queries are ignored. Multiple queries
may be presence-positive, while the production classification head remains
strict one-to-one.

## Held-out LSP evaluation

The fold3 dense GT was evaluated with raw, NMS-free argmax predictions. Values
below were rescored on 2026-10-25 with PIL polygon filling and LSP-DETR's
distance-based overlap ownership. Train36 stopped at epoch 246, so it is a
provisional, unmatched reference for these epoch-332 runs.

| Metric | Train31 | Train30 | Train36 |
|---|---:|---:|---:|
| Whole bPQ | 0.6269 | 0.6274 | 0.6274 |
| Whole bDQ | 0.7678 | 0.7666 | 0.7690 |
| Whole bSQ | 0.8006 | 0.8023 | 0.8007 |
| Whole mPQ | 0.4434 | 0.4404 | 0.4445 |
| Macro bPQ | 0.6446 | 0.6462 | 0.6458 |
| Macro mPQ | 0.4458 | 0.4444 | 0.4498 |
| Centroid precision | 0.8584 | 0.8546 | 0.8555 |
| Centroid recall | 0.7442 | 0.7440 | 0.7463 |
| Raw predictions | 57,784 | 58,025 | 58,149 |

The changes are within run-to-run noise. Presence slightly raises precision
and whole-fold mPQ relative to Train30, but does not improve bPQ or recall and
does not beat Train36 detection quality. This auxiliary target did not resolve
foreground suppression.

## Next controlled variant

Return to the Train30 branch (`detr_presence_weight: 0.0`) and add owner-only
appearance exemplar imitation. Small Hungarian owners (GT area at most 200
pixels) are students. Correctly fired larger same-type owners provide detached
exemplars, and unmatched mutual-center nucleus-like queries provide a detached
duplicate-negative prototype. Matching, production losses, and inference are
unchanged.

Launch every matched variant with `--epochs 600 --end 332` so the learning-rate
schedule and stopping point remain identical.
