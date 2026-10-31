# V2 Train30 — supervised query appearance

- Output: `output/v2/train30`
- Checkpoint evaluated: `weights/last.pt`
- Stop epoch: **332** of the unchanged 600-epoch cosine schedule.

## Change from Train36

After layer-3 query recasting, a 32-channel stride-2 trainable appearance map
from detection-stream `b1` and `b3` is sampled at the query center, eight
interior radial points, and eight exterior radial points. The summaries enter
the query through a zero-initialized residual projection before layers 4–6.
The map receives union-nucleus BCE + Dice supervision with weight 0.25.
Polygon/oracle masked attention is disabled. The auxiliary presence head is
implemented but disabled (`detr_presence_weight: 0.0`).

## Held-out LSP evaluation

Command used the fold3 dense GT and `eval_lsp.py` with raw, NMS-free argmax
predictions. Values below were rescored on 2026-10-25 with PIL polygon filling
and LSP-DETR's distance-based overlap ownership. Train36 stopped at epoch 246,
so it is a provisional, unmatched reference for this epoch-332 run.

| Metric | V2 Train30 | Train36 |
|---|---:|---:|
| Whole bPQ | 0.6274 | 0.6274 |
| Whole bDQ | 0.7666 | 0.7690 |
| Whole bSQ | 0.8023 | 0.8007 |
| Whole mPQ | 0.4404 | 0.4445 |
| Macro bPQ | 0.6462 | 0.6458 |
| Macro mPQ | 0.4444 | 0.4498 |
| Centroid precision | 0.8546 | 0.8555 |
| Centroid recall | 0.7440 | 0.7463 |

The appearance path produced a small bSQ improvement but did not improve
detection quality or macro type quality enough to separate from Train36.

## Matched-stop protocol

For every subsequent controlled variant of this experiment, launch with
`--epochs 600 --end 332`. The trainer preserves the original 600-epoch
learning-rate schedule, completes validation and checkpoint saving for epoch
332, and then stops cleanly. Do not launch with `--epochs 332`.
