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
predictions.

| Metric | V2 Train30 | Train36 |
|---|---:|---:|
| Whole bPQ | 0.6273 | 0.6269 |
| Whole bDQ | 0.7663 | 0.7685 |
| Whole bSQ | 0.8025 | 0.8006 |
| Whole mPQ | 0.4405 | 0.4443 |
| Macro bPQ | 0.6461 | 0.6449 |
| Macro mPQ | 0.4445 | 0.4493 |
| Precision | 0.8550 | 0.8554 |
| Recall | 0.7443 | 0.7463 |

The appearance path produced a small bSQ improvement but did not improve
detection quality or macro type quality enough to separate from Train36.

## Matched-stop protocol

For every subsequent controlled variant of this experiment, launch with
`--epochs 600 --end 332`. The trainer preserves the original 600-epoch
learning-rate schedule, completes validation and checkpoint saving for epoch
332, and then stops cleanly. Do not launch with `--epochs 332`.
