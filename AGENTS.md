# AGENTS.md — RayCastED current working instructions

## Project state — 2026-10-26

This repository is the **RayCastED** project on branch
`feature/nulite-raycast-seg`. The old YOLO26-FCN material is historical only;
do not use its architecture, loss, or metric guidance for current work.

The objective is PanNuke nuclei detection without inference-time NMS,
clustering, confidence-threshold tuning, or post-hoc deduplication. The model
must make the foreground/empty and ownership decisions directly through its
one-to-one query head.

Read `HANDOVER.md` before changing model, loss, configuration, evaluation, or
training plans. It is the current experiment record.

## Boundaries

- Do not edit code, configuration, documentation, checkpoints, or active-run
  files unless the user explicitly requests a change.
- Do not launch, stop, resume, or interrupt training. The user controls
  training. Frozen-checkpoint evaluation and read-only diagnostics are allowed
  when requested and when the GPU is idle.
- Preserve the NMS-free inference contract. Polygon overlap ownership inside
  `eval_lsp.py` is evaluator-only mask construction, not inference
  post-processing.
- Do not treat validator PQ, historical FCN results, or contaminated historical
  NuLite measurements as literature-comparable.

## Current architecture

NuLite-DETR is the current query-based **version of RayCastED**, not a separate
project or model line. The baseline (Train50) is a thin model: a frozen
NuLite-T8 U-Net produces the foreground map and also supplies `b3`/`b4`/`b5` to
the DETR decoder through depthwise-pointwise fusion adapters, and the oracle is
thawed from epoch 50 except for `np_head`. The decoder predicts a centroid plus
64 rays and independent sigmoid logits for five nucleus classes plus explicit
no-object. Foreground at inference is the raw all-class argmax: a query is
retained only when a real class beats the empty column.

The head uses 400 fixed grid queries, class-aware Hungarian one-to-one
matching, and `cost_inside`. Five decoder layers share three distinct
transformer weights (`detr_shared_layer_groups: 3`) at `hd` 192 / `d_ffn` 512.
Main implementation files:

- `raycasted/model/nulite_detr_model.py` — model, loss composition, oracle and
  auxiliary objectives.
- `raycasted/model/nulite_detr_loss.py` — Hungarian matcher and polygon loss.
- `raycasted/model/blocks/nulite_detr.py` — query decoder and recasting.
- `raycasted/model/blocks/nulite.py` — NuLite encoder-decoder and oracle.
- `raycasted/model/train.py` — config wiring, data collation, validator labels.
- `raycasted/model/train_nulite_seg.py` — standalone oracle trainer.
- `main/pannuke-thin-t8.yaml` — the current baseline recipe.
- `scripts/train_thin_t8.sh` — two-stage seg-then-det runner.

## Evaluation protocol

`raycasted/evals/eval_lsp.py` is the sole literature-facing evaluator.

- Use original dense PanNuke masks from the held-out fold parquet.
- Keep raw query outputs using the `nc+1` all-class argmax; apply no confidence
  threshold and no NMS.
- Rasterize polygons with PIL. Resolve overlap pixels by greatest Euclidean
  interior distance, matching LSP-DETR test-time mask ownership.
- Report per-image bPQ/mPQ then their means, plus per-tissue macro values over
  all 19 tissues. Use final `last.pt` for a completed run.

`eval_pannuke.py` and historical output documents use other protocols. They may
help compare an internal training trajectory, but cannot support claims against
LSP-DETR or LKCell. Details and the evaluator correction are in
`docs/diag/lsp-evaluator-correction-2026-10-25.md`.

## Data and cross-validation

PanNuke folds are fold1=2,656, fold2=2,523, fold3=2,722 tiles; classes are
0–4 and there are 19 tissue types. Dense GT parquets are in
`raycasted/data/dataset/PanNuke/data/`.

For a literature result, train one model per fold rotation, evaluate its
genuinely held-out fold with `eval_lsp.py`, and report mean plus standard
deviation across the three final checkpoints. Existing single-fold results are
useful diagnostics, not a paper claim.

## Current evidence and baseline

Corrected held-out Fold3 evaluation with `eval_lsp.py`, raw `nc+1` argmax, no
NMS (2,722 images, 66,654 instances):

| Checkpoint | Params | bPQ | bDQ | bSQ | mPQ | Macro bPQ | Macro mPQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`output/v2/train50_thin-t8-aggressive/weights/last.pt`** | **6.16M** | 0.6337 | 0.7755 | 0.8025 | 0.4546 | 0.6532 | 0.4609 |
| `output/v2/train44_frozen-thaw-full/weights/last.pt` | 15.27M | 0.6374 | 0.7795 | 0.8044 | 0.4544 | 0.6578 | 0.4617 |
| `output/v2/train34_viable-owner-foreground/weights/last.pt` | 24.7M | 0.6325 | 0.7759 | 0.8009 | 0.4476 | 0.6521 | 0.4578 |
| `output/f1-train/train36/weights/last.pt` | 24.7M | 0.6274 | 0.7690 | 0.8007 | 0.4445 | 0.6458 | 0.4498 |

**Train50 is the baseline**: a thin FastViT-T8 model at 6.16M parameters,
configured by `main/pannuke-thin-t8.yaml` and documented in
`docs/runs/nulite-detr/v2-train50-thin-t8.md`. It keeps the frozen-oracle
architecture of Train44 (frozen NuLite U-Net + depthwise-pointwise fusion
adapters + full U-Net thaw at epoch 50) and thins only the backbone and the
decoder: `fastvit_t8`, `detr_ndl: 5`, `detr_hd: 192`, `detr_d_ffn: 512`,
`detr_shared_layer_groups: 3`.

Train36 stopped at epoch 246; Train30/31/34/44/50 stopped at epoch 332.
Train36 is therefore a provisional, unmatched reference.

The dominant held-out failure is not duplicate ownership: many small nuclei
already have a good raw polygon, but their query stays empty. On Train50 the
`good_suppressed_bg` cohort is 8,782 nuclei (down 949 from Train44) while
`unmatched_predictions` rose by 1,780, so thinning trades precision for recall.
A balanced linear probe on the exact proposal features reaches AUC 0.912
(0.892 size-controlled) against a trained head operating at foreground recall
0.650, so the binding constraint is the one-to-one Hungarian decision rule, not
representation or capacity.

## Working rules for new ablations

- Start from an explicit control configuration and change one mechanism.
- Keep class-aware Hungarian matching, raw argmax inference, and no NMS fixed
  unless the experiment explicitly tests one of those contracts.
- Use `--epochs 600 --end 332` for this current controlled series so the
  learning-rate schedule and stop point match the Train30/31/34/44/50 series.
- Add focused loss/gradient/config tests. Run the NuLite integration tests,
  scoped Ruff, and `git diff --check` before handing off a code change.
- Evaluate a completed checkpoint on the genuine held-out fold before drawing
  conclusions. Break outcomes down by good-polygon/background suppression,
  no-good-polygon misses, size, crowding, type, and duplicates.

Historical FCN and pre-correction NuLite records remain in git history and old
run documents for provenance only.
