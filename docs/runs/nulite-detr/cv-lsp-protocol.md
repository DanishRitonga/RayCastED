# LSP-DETR-protocol eval: first clean numbers

Date: 2026-09-07. Tool: `raycasted/evals/eval_lsp.py` (new, committed with this doc).

Correction (2026-10-25): the original evaluator used OpenCV polygon filling and
largest-area-first overlap ownership, whereas LSP-DETR uses PIL filling and
per-pixel interior-distance ownership. The evaluator now follows the latter
and never applies polygon NMS. Numbers in this historical document were
computed before that correction; see `docs/diag/lsp-evaluator-correction-2026-10-25.md`
for corrected held-out results and the measured effect of the protocol change.

## Why this eval exists

All historical "test-set" numbers (train18-36) were measured on fold3 tiles that were
the *training* fold (contamination discovered when rotation-1 of the 3-fold CV was
built). To compare honestly against LSP-DETR / LKCell we replicate their exact eval
protocol:

- GT = original dense PanNuke masks from the fold parquet (not 64-ray polygons).
- No confidence threshold — every query whose argmax is a real class (∅ dropped) is kept.
- Per-image binary PQ (IoU>0.5, Hungarian matching) then **mean over images**; same for
  per-class mPQ (mean over GT-present classes per image); masked (bMPQ/mMPQ) variants.
- Centroid F1 (radius 12 px), global tp/fp/fn.
- Per-tissue accumulation; the reference Table-3 row is the macro mean over the 19 tissues.

## Results (held-out folds, clean)

### Rotation-1 — locked t12-sep recipe (fold1 train → fold3 test)
weights: `output/f1-train/train/weights/best.pt` (== last.pt, plateau at ep600)
model: NuLiteRayCastDETRModel, fastvit_t12 + separable decoder + shared layers + hd128,
7.85M params, 400 grid queries, no_object ∅ class.

GT instances: 66,654. Pred instances (all non-∅): 51,506.

| metric | whole-fold | macro over 19 tissues |
|---|---|---|
| bPQ | 0.5969 | 0.6202 |
| mPQ | 0.4190 | 0.4293 |
| bDQ | 0.7369 | — |
| bSQ | 0.7919 | — |
| bMPQ | 0.6082 | — |
| mMPQ | 0.4261 | — |
| F1 (r=12) | 0.7736 (P 0.8873 / R 0.6857) | — |

Tissue range: bPQ Colon 0.4948 (worst) → Kidney 0.6661 (best); mPQ Colon 0.3535 →
Bladder 0.5126.

### train19 recipe — s12 OLD (pre-parity) decoder (fold3 train → fold2 test)
weights: `output/nulite/train19/weights/best.pt` (15.46M, MDS off, S+F+Q, no
per-layer-match). Evaluated under era-correct code (git worktree b7444b6).

GT instances: 59,872. Pred instances (all non-∅): 46,415.

| metric | whole-fold | macro over 19 tissues |
|---|---|---|
| bPQ | 0.5973 | 0.6148 |
| mPQ | 0.4200 | 0.4223 |
| bDQ | 0.7349 | — |
| bSQ | 0.7938 | — |
| bMPQ | 0.6094 | — |
| mMPQ | 0.4277 | — |
| F1 (r=12) | 0.7778 (P 0.8905 / R 0.6904) | — |

## Interpretation vs LSP-DETR reference

LSP-DETR Table-3 macro average (3-fold CV, final ckpt): **bPQ 67.5 / mPQ 48.2**
(LKCell 68.4/50.3, CellViT 67.9/49.8).

- Our two best clean configs land at macro bPQ ~0.61-0.62 / mPQ ~0.42-0.43 on a single
  held-out fold — a genuine gap of ~5-6 bPQ pts and ~5-6 mPQ pts below LSP-DETR.
- Caveats in our favor: single fold (not 3-fold mean), best.pt (not final ckpt) — but
  best==last here, so ckpt choice is moot for rotation-1.
- The gap is dominated by **recall** (F1 0.77, R ~0.69, P ~0.89): the model finds ~2/3
  of nuclei at the ∅-argmax decision boundary; precision is high. mPQ drag is consistent
  with per-class recall on Necrosis/dead (small, sparse class).

## Next steps (candidate)

1. Finish rotations 2 and 3 of the locked recipe for a 3-fold CV mean±std.
2. Raise recall: no-object ∅ weight (currently 2.0) / inference decision (∅ argmax is
   strict); evaluate a conf-percentile or ∅-prob threshold sweep under this protocol.
3. Revisit architecture levers that clean evals now show matter (old vs parity decoder
   choice is ~wash under this protocol: 0.620 vs 0.615 on different folds).

## Files

- `raycasted/evals/eval_lsp.py` — protocol eval (mirrors LSP-DETR metrics source).
- Raw logs: `/tmp/opencode/eval_lsp_f1train.log` (rotation-1 best.pt), `/tmp/opencode/eval_lsp_f1train_last.log`, train19 fold2 run captured in conversation.
