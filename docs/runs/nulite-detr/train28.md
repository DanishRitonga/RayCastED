# Train28 — s12 full-seed + per-layer matching (LSP-DETR look-forward-twice)

## Run
- Date: 2026-08-31
- Branch: `feature/nulite-raycast-seg` (HEAD `8bf7036`)
- Output: `output/nulite/train28` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs, best = ep600)
- Config: train19 S+F+Q base (`lambda_seg: 1.0`, `detr_seed_feature: true`, `detr_seed_in_content: true`, `detr_no_object_weight: 2.0`, `detr_cost_inside: 10.0`, `detr_mds: false`, grid queries, 3 layers hd=256, `nulite_variant: fastvit_s12`) **+ `detr_per_layer_match: true`** — the ONLY change from train19. Also restored `nulite_variant: fastvit_s12` after the T8 (train27) experiment and lowered `patience` to 300.

## Hypothesis
LSP-DETR matches each decoder layer independently ("look forward twice" as in DINO [41]); our `_get_loss_aux` reused the last-layer Hungarian match for all aux layers. Per-layer matching should give layers 0/1 cleaner gradient targets → better refinement → better winner-selection (the confirmed bottleneck). Training-only change: no params or inference impact.

## Results

### Val (train set; still climbing at ep600)
Best fitness ep599 (0.552): mAP50 0.45097, bPQ 0.59534; best bPQ ep600 0.59538 (bSQ 0.7791, bDQ 0.72475); best mAP50 ep595 0.45107. ≈ train19 val (bPQ 0.5957 / mAP50 0.4462) — val is a wash.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
AJI 0.7514, AP@0.5 0.8820, AP@0.7 0.6840, AP@0.9 0.0192, AP@0.5:0.05:0.95 0.5072, bPQ 0.6800, bMPQ 0.7783, mPQ 0.6442, mMPQ 0.7496, F1(centroid, r=12) 0.8878, Prec 0.8065, Rec 0.9872, Params 15.46M, 5.55 ms/img, **80,596 preds (1.22x)**, matched 65,003/65,848.

Class centroid F1: Neoplastic 0.8730 / Inflammatory 0.9154 / Connective 0.8797 / Necrosis 0.7956 / Epithelial 0.8999. Size recall: small 0.9672 / medium 0.9959 / large 0.9981. Tissue macro: bPQ 0.6875 ± 0.0485, mPQ 0.7135 ± 0.0380, AJI 0.7642 ± 0.0498 (worst Colorectal bPQ 0.5774; best Thyroid 0.7419). Polygon-level (area-ratio metric, relative only): bPQ 0.8477, centroid L2 83.61px, ray L1 1.66px. Unmatched GT: 845 (601 truly missed >12px, 243 wrong-class within 12px).

**vs train19 (shared match):**

| metric | train19 | train28 | Δ |
|---|---|---|---|
| bPQ | 0.6728 | **0.6800** | +0.007 |
| mPQ | 0.6294 | **0.6442** | +0.015 |
| AP@0.5 | 0.8758 | **0.8820** | +0.006 |
| AP@50-95 | 0.5023 | **0.5072** | +0.005 |
| AJI | 0.7521 | 0.7514 | ~0 |
| F1 | 0.8823 | **0.8878** | +0.006 |
| Prec | 0.7937 | **0.8065** | +0.013 |
| Rec | 0.9933 | 0.9872 | −0.006 |
| preds | 82,409 (1.25x) | **80,596 (1.22x)** | fewer |
| ms/img | 5.10 | 5.55 | +0.45 |

## Diagnosis
**Per-layer matching is a genuine small win, not a wash.** bPQ +0.7pt, mPQ +1.5pt, precision +1.3pt, fewer duplicates (1.22x vs 1.25x preds) — the cleaner aux gradients improved winner-selection exactly as the LFT hypothesis predicted. Only costs: recall −0.6pt and +0.45 ms/img. This is consistent with the decoder/ndl ablation pattern: layer refinement quality is the bottleneck, and per-layer matching feeds it better. **train28 (s12 + S+F+Q + per-layer match) is the new best and the submission candidate.**

## Exact original-mask eval (literature protocol)
Via `/tmp/opencode/orig_mask_eval.py` (CKPT repointed to train28 best.pt), fold3, conf=0.20, identical pred masks in both columns:

| metric | polygon-GT (current) | **original-mask GT** |
|---|---|---|
| bPQ (DQ/SQ) | 0.6800 (0.8397/0.8098) | **0.7013 (0.8386/0.8362)** |
| mPQ | 0.6442 | **0.6687** |
| AJI | 0.7514 | **0.7738** |
| centroid F1 (P/R) | 0.8878 (0.8065/0.9872) | **0.8799 (0.8038/0.9719)** |
| binary recall | 0.9337 (61,484/65,848) | 0.9263 (61,743/66,654) |

Per-class PQ (original-mask GT): Neoplastic 0.7037, Inflammatory 0.7239, Connective 0.6828, Necrosis 0.5185, Epithelial 0.7147.

**Vs literature (3-fold CV + watershed protocol)**: LSP-DETR bPQ 0.675/mPQ 0.482 (45M), LKCell bPQ 0.684/mPQ 0.503 (163.8M). **train28 single-fold bPQ 0.7013 / mPQ 0.6687 at 15.46M beats both on bPQ AND mPQ** — the first run to cross 0.70 bPQ. Necrosis remains the weak class (PQ 0.52). Caveats unchanged: single fold (fold3), largest-first overlap resolution vs watershed, no 3-fold CV averaging.

## Next
- Update `docs/paper/benchmark.md` with the train28 row (new best).
- Paper Pareto axis candidates remaining: head-slimming C (hd 128 → 13.04M) / D (shared layers → 13.47M) for a mid-size point.
