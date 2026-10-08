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

The fold3 dense GT was evaluated with raw, NMS-free argmax predictions.

| Metric | Train31 | Train30 | Train36 |
|---|---:|---:|---:|
| Whole bPQ | 0.6266 | 0.6273 | 0.6269 |
| Whole bDQ | 0.7670 | 0.7663 | 0.7685 |
| Whole bSQ | 0.8009 | 0.8025 | 0.8006 |
| Whole mPQ | 0.4432 | 0.4405 | 0.4443 |
| Macro bPQ | 0.6442 | 0.6461 | 0.6449 |
| Macro mPQ | 0.4454 | 0.4445 | 0.4493 |
| Precision | 0.8582 | 0.8550 | 0.8554 |
| Recall | 0.7439 | 0.7443 | 0.7463 |
| Raw predictions | 57,784 | 58,021 | — |

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
