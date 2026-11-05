# Next training on this device: T12 decoder grid

Date: 2026-11-04. Status: **plan — not launched**. The user controls training.

The T8 decoder grid finished overnight and has been evaluated on held-out Fold3.
The T12 grid was written and committed alongside it but **never ran**: the
device spent the whole night on `train53_ablation-*`. That grid is the next
training job.

## Why this is next

The T8 grid answers whether the head thinning baked into Train50 is load-bearing
for a small backbone. It does not answer the same question for T12, which is the
backbone that wins whole-fold bPQ in the FastViT sweep (0.6395 bPQ) but loses on
macro bPQ, and which is the closest thing to a "larger model" option if hd128 at
T8 turns out to be too weak. The T12 grid is the symmetric experiment: the same
six one-key deltas, the same epoch schedule, the same oracle.

It also matters for the Pareto argument. T8 finished with nDL depth and hidden
width both ~flat, so the head config chosen at T8 will look arbitrary unless the
same sweep at a larger backbone shows the same flatness. If T12 instead rewards
more capacity (`ndl6`, `hd256`), the thinning claim becomes backbone-specific,
which is a result worth knowing before the 3-fold rotation is spent.

## What exists already

- Oracle: `output/oracle/nuLiteT-fold1-sep-t12.pt` (no seg stage needed).
- Base run: `output/v2/train52_fastvit-t12` (70.9 s/epoch).
- Recipes: `main/pannuke-thin-t12.yaml` plus six one-key variants.
- Runner: `scripts/train_ablation_t12.sh`.

Each recipe was verified with `ETLConfig` to differ from the base in exactly one
key and nothing else:

| recipe | key | base → variant |
|---|---|---|
| `main/pannuke-thin-t12-ndl6.yaml` | `detr_ndl` | 5 → 6 |
| `main/pannuke-thin-t12-ndl4.yaml` | `detr_ndl` | 5 → 4 |
| `main/pannuke-thin-t12-g2.yaml` | `detr_shared_layer_groups` | 3 → 2 |
| `main/pannuke-thin-t12-g0.yaml` | `detr_shared_layer_groups` | 3 → 0 |
| `main/pannuke-thin-t12-hd128.yaml` | `detr_hd` | 192 → 128 |
| `main/pannuke-thin-t12-hd256.yaml` | `detr_hd` | 192 → 256 |

## Command

```bash
scripts/train_ablation_t12.sh
```

Runs all six sequentially as `train55_ablation-t12-<v>`, `--epochs 600
--end 332`, batch 16, and skips any config whose `last.pt` already exists, so
re-invoking after an interruption resumes the queue. Overridable:
`VARIANTS`, `DEVICE`, `WORKERS`, `BATCH`, `OUTPUT`, `LOG_DIR`, `DET_EPOCHS`,
`DET_END`.

## Expected cost

T12 base is 70.9 s/epoch versus 59.6–69.5 s/epoch for the T8 grid, so scaling
each T8 wall time by T12/T8:

| config | T8 wall | T12 estimate |
|---|---:|---:|
| ndl6 | 6.43 h | ~7.1 h |
| ndl4 | 5.51 h | ~6.1 h |
| g2 | 5.96 h | ~6.6 h |
| g0 | 5.99 h | ~6.6 h |
| hd128 | 5.54 h | ~6.1 h |
| hd256 | 6.32 h | ~7.0 h |
| **total** | **35.7 h** | **~39–40 h** |

Approximately one and a half days of continuous device time.

## Evaluation afterwards

Same protocol as the T8 grid: `eval_lsp.py`, held-out Fold3 parquet, raw `nc+1`
argmax, no NMS, `last.pt`, `--save-predictions` into a tmp cache for paired
cohort audits.

```bash
for v in ndl6 ndl4 g2 g0 hd128 hd256; do
  .venv/bin/python -m raycasted.evals.eval_lsp \
    --weights output/v2/train55_ablation-t12-$v/weights/last.pt \
    --data-dir output/v2/transformed/test \
    --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
    --batch 8 --device 0 --workers 4 \
    --save-predictions /tmp/opencode/ablation_eval/pred_t55_$v.npz
done
```

Roughly 2.5 min per checkpoint on this device.

## Decision rules, fixed in advance

- Compare against the T12 base (`train52_fastvit-t12`, bPQ 0.6372, macro bPQ
  0.6465) as well as against the T8 winner (hd128, bPQ 0.6333, 5.77M).
- A config only counts as an improvement if it moves **both** whole-fold bPQ and
  macro bPQ up by more than the ~0.002 run-to-run noise seen in the T8 grid.
- `g2` losing again would confirm layer sharing is the one lever that reliably
  hurts. `hd256` or `ndl6` winning at T12 where they were flat at T8 would make
  the thinning backbone-specific and must be reported as such.
- No held-out claim is paper-grade until the 3-fold rotation exists; every
  oracle on disk is fold1-only, so rotations 2 and 3 still need their own
  oracles and transformed trees.

## Prerequisite check before launch

The device was rebooted after a driver/userspace mismatch (now both 595.99.02,
`nvidia-smi` healthy). `/tmp` is tmpfs, so the T8 prediction caches in
`/tmp/opencode/ablation_eval/` did not survive the reboot; re-run those evals
before paired cohort comparisons against the T12 grid.
