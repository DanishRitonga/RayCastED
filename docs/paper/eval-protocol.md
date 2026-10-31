# PanNuke Evaluation Protocol (literature-comparable)

This is the **only** evaluation that can be compared against published PanNuke
numbers (LSP-DETR, LKCell, CellViT, …). Read this before reporting any metric.

## TL;DR

- Use `raycasted/evals/eval_lsp.py`, on a **genuinely held-out fold**, with
  **no confidence threshold**, `argmax != no-object` decoding.
- Report **per-tissue macro bPQ / mPQ** (the `Average` row), plus whole-fold
  per-image means for detail.
- Report the **final/last** checkpoint (see "Checkpoint choice" below).
- Anything else (`eval_pannuke.py`, the validator's fitness, `docs/runs/*`
  historical numbers) is **not** comparable to the literature.

## Why other numbers don't count

| source | protocol | comparable to literature? |
|---|---|---|
| `evals/eval_lsp.py` | dense parquet GT, no conf threshold, per-image PQ, per-tissue macro | **YES** |
| `evals/eval_pannuke.py` | older global-PQ protocol, `conf=0.20`, polygon GT | no |
| validator `fitness` (0.3·mAP50 + 0.7·bPQ) | polar same-center IoU proxy @ `conf=0.20`, on the **val** fold | no — used only for checkpoint selection |
| `docs/runs/nulite-detr/*.md`, `docs/paper/benchmark.md` (historical) | training-fold-contaminated | no — do not cite |

Contamination rule: a model may only be evaluated on a fold it was **not
trained or validated on**. Historical `train18..36` "test" numbers were measured
on their training fold and are therefore invalid for comparison.

## Data

- Folds (HF `RationAI/PanNuke`): **fold1 = 2656**, **fold2 = 2523**,
  **fold3 = 2722** tiles, 256×256, 0.25 mpp.
- GT: dense instance masks in the fold parquet
  (`raycasted/data/dataset/PanNuke/data/fold<N>-00000-of-00001.parquet`),
  categories `0..4` = Neoplastic, Inflammatory, Connective, Dead, Epithelial;
  **19 tissues** (per-tissue macro is taken over these).
- Tile ↔ GT matching: tiles are `*_roi_<id>.npz`; the ROI id indexes the
  parquet row (verified: ids unique, tile `tissue` field matches parquet row).
- Tile trees (rotation-1, `output/f1-train/transformed/`):
  `train/` = fold1, `val/` = fold2, `test/` = fold3.

## Decoding (must match training)

1. Run inference on each held-out tile (no augmentation).
2. Over the `nc + 1` logits (last column = no-object ∅): **keep** a query iff
   `argmax(logits) != last column`.
3. Class label = `argmax` over the real columns; **no confidence threshold** is
   applied — a query is kept purely by the argmax decision.
4. Predicted polygons are normalized → pixels via `× crop_size` (256).
5. Fill each polygon with PIL, then assign overlapping pixels to the mask
   with the greatest Euclidean interior distance, as in LSP-DETR. Remove
   predictions left empty by this assignment. There is no NMS or other
   duplicate filter. The previous largest-area-first resolver is available
   only through `--compare-legacy-overlap` for a labeled diagnostic.

## Metrics

- **bPQ / bDQ / bSQ**: per-image binary PQ, then mean over images.
  Per image: IoU matrix (preds × targets), Hungarian on `−IoU`, keep matches
  with `IoU > 0.5`; `tp = #kept`, `dq = tp / (0.5·P + 0.5·M)`,
  `sq = Σ matched IoU / tp`, `pq = dq·sq`. Images with empty GT are skipped.
- **mPQ / mDQ / mSQ**: per-image mean over GT-present classes, then mean over
  images (captures classification/type errors that bPQ hides).
- **bMPQ / mMPQ**: masked variants — pred pixels overlapping any GT are zeroed
  first (isolates false positives on background).
- **F1 / P / R**: centroid-distance Hungarian with **radius 12 px**, global
  tp/fp/fn sums.
- **`Average` row = per-tissue macro** over the 19 tissue means. This is the
  headline number to report.

## Command

```bash
uv run python -m raycasted.evals.eval_lsp \
  --weights output/f1-train/train<N>/weights/last.pt \
  --data-dir output/f1-train/transformed/test \
  --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
  --batch 16 --device 0
# optional: --workers 4 ; --compare-legacy-overlap (historical comparison)
```

Logs are typically written to `/tmp/opencode/eval_lsp_train<N>_fold3.log`.

## Checkpoint choice

- The reported number is for the **final** checkpoint (`last.pt`); for
  rotation-1 `best.pt == last.pt`.
- **Caution:** `best.pt` is chosen by the validator's proxy fitness, not by
  `eval_lsp`. The proxy (polar same-center IoU + `conf=0.20`, on the val fold)
  can mis-rank runs (measured bias +0.003…+0.030, and it has reversed orderings
  vs the real metric). Prefer reporting `eval_lsp` on `last.pt`, or report both.
- Within a run, evaluation uses the **EMA** weights (intended).

## 3-fold cross-validation

1. Pick a rotation: `split_map` in `main/pannuke.yaml` assigns
   train/val/test folds (rotation-1: train=fold1, val=fold2, test=fold3).
2. Train one model per rotation with a **fresh `--output` dir**.
3. Evaluate the held-out test fold with `eval_lsp.py`.
4. Report **mean ± std of the per-tissue macro bPQ / mPQ over the 3 test folds**,
   using the final checkpoint.

```bash
uv run python -m raycasted.pipeline --config main/pannuke.yaml \
  --output output/<run> --stage train --epochs 600 --batch 16 --device 0
```

## Reference (per-tissue macro, 3-fold CV, final ckpt)

| model | bPQ | mPQ |
|---|---|---|
| LSP-DETR | 0.675 | 0.482 |
| LKCell | 0.684 | 0.503 |
| CellViT | 0.679 | 0.498 |

Historical held-out fold3 results (all trained on fold1), for orientation.
These were scored before the overlap correction, so use the corrected
evaluator for any new comparison:

| run | macro bPQ | macro mPQ | note |
|---|---|---|---|
| train23 | **0.6393** | 0.4369 | best bPQ (grid + seed-as-4th-level) |
| train18 | 0.6385 | 0.4369 | capacity head |
| train19 | 0.6362 | **0.4417** | best mPQ (round-robin schedule) |
| train26 | 0.6289 | 0.4367 | best bDQ 0.7572 |

## Gotchas

- Never evaluate on the training or validation fold.
- Do not compare validator `fitness` / `bPQ` across runs — different protocol,
  and it is noisy/bias-prone.
- `eval_pannuke.py` numbers are historical and not comparable; keep it for
  internal deltas only.
- Empty-GT images are skipped in the PQ mean (both protocols), so the
  denominator is the number of non-empty images.
- Evaluations recorded before the 2026-10-25 overlap correction used the
  largest-first rule; see `docs/diag/lsp-evaluator-correction-2026-10-25.md`.
