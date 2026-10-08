# V2 Train32 — supervised appearance + owner exemplar imitation

> **Protocol note:** this run is outside the three-run supervised query
> appearance experiment. That plan contains only a fresh control, appearance,
> and appearance + presence. Train32 should be treated as a separate follow-up.

- Output: `output/v2/train32`
- Checkpoint evaluated: `weights/last.pt`
- Stop epoch: **332** of the unchanged 600-epoch cosine schedule.

## Change from Train30

Train32 retains Train30's supervised stride-2 appearance map and query-local
polygon sampling. The Train31 presence objective is disabled. A training-only
cosine-softmax loss pulls small Hungarian-owner appearance embeddings (GT area
at most 200 pixels) toward detached, correctly fired larger owners of the same
nucleus type. Unmatched mutual-center nucleus-like queries contribute a
detached duplicate-negative prototype. Its loss weight is 0.1 and temperature
is 0.1. Production matching and inference remain strict one-to-one and NMS-free.

## Held-out LSP evaluation

| Metric | Train32 | Train30 | Train31 | Train36 |
|---|---:|---:|---:|---:|
| Whole bPQ | 0.6272 | 0.6273 | 0.6266 | 0.6269 |
| Whole bDQ | 0.7666 | 0.7663 | 0.7670 | 0.7685 |
| Whole bSQ | 0.8013 | 0.8025 | 0.8009 | 0.8006 |
| Whole mPQ | 0.4435 | 0.4405 | 0.4432 | 0.4443 |
| Macro bPQ | 0.6468 | 0.6461 | 0.6442 | 0.6449 |
| Macro mPQ | 0.4478 | 0.4445 | 0.4454 | 0.4493 |
| Precision | 0.8609 | 0.8550 | 0.8582 | 0.8554 |
| Recall | 0.7387 | 0.7443 | 0.7439 | 0.7463 |
| Raw predictions | 57,195 | 58,021 | 57,784 | — |

Exemplar imitation shifts the operating point toward precision by roughly
0.006 and away from recall by roughly 0.006 relative to Train30. Detection PQ
and DQ are unchanged within run noise. It does not justify combining imitation
with additional supervision.

Train32 completes the three-run appearance plan. Since neither auxiliary
appearance/presence supervision nor exemplar imitation produced a meaningful
held-out detection gain, there is no fourth controlled variant in this plan.
