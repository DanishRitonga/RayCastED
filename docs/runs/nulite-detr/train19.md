# Train19 — full-seed baseline, MDS OFF (S+F+Q)

## Run
- Date: 2026-08-28
- Branch: `feature/nulite-raycast-seg` (HEAD `292769a`)
- Output: `output/nulite/train19` (600 epochs, amp=false, patience=500, plain cosine, 0 zero-metric epochs)
- Config change from train18: `detr_mds: true → false` (MDS-DETR rank-causal self-attn mask **off** — it proved redundant; duplicate suppression comes from cost_inside + FCN-parity loss). Full seed branch: `lambda_seg: 1.0`, `detr_seed_feature: true`, `detr_seed_in_content: true` (S+F+Q). `detr_cost_inside: 10.0`, `detr_no_object_weight: 2.0`, grid queries, 3 layers hd=256.

## Hypothesis
MDS off is a pure ablation of the mask (train18 MDS on = bPQ 0.6713). If bPQ ≈ train18 → MDS redundant, strip it. This config (S+F+Q, no MDS) is also the intended **seed-arc winner** and the base for the head-slimming ladder.

## Results

### Val (train set; still climbing at ep600)
Best fitness ep600 (0.5508): bPQ 0.5957, bSQ 0.7825, bDQ 0.7219, mAP50 0.4462, mAP50-95 0.3375, P 0.4480, R 0.6591.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
AJI 0.7521, AP@0.5 0.8758, AP@0.7 0.6754, AP@0.9 0.0223, AP@0.5:0.05:0.95 0.5023, bPQ 0.6728, bMPQ 0.7744, mPQ 0.6294, mMPQ 0.7366, F1(centroid, r=12) 0.8823, Prec 0.7937, Rec 0.9933, Params 15.46M, 5.10 ms/img, **82,409 preds (1.25x)**, matched 65,285/65,848 (overall recall 65,407/65,848 = 0.9933).

Class centroid F1: Neoplastic 0.8667 / Inflammatory 0.9134 / Connective 0.8738 / Necrosis 0.7508 / Epithelial 0.8917. Size recall: small 0.9826 / medium 0.9980 / large 0.9992. Tissue macro: bPQ 0.6842 ± 0.0486, mPQ 0.7069 ± 0.0487, AJI 0.7663 ± 0.0501 (worst Colorectal bPQ 0.5633; best Liver 0.7369 / Thyroid 0.7481). Polygon-level (area-ratio metric, relative only): bPQ 0.8402.

## Diagnosis
**MDS is redundant** — train19 (off) within noise of train18 (on): bPQ 0.6728 vs 0.6713, mPQ 0.6294 vs 0.6321, ~10% faster (5.10 vs 5.70 ms/img). Duplicate suppression is carried by cost_inside + FCN-parity loss + the 1:1 Hungarian head, not the rank-causal mask. `detr_mds: false` becomes the standard. **train19 is the seed-arc winner (S+F+Q)** and the head-ladder base.

## Exact original-mask eval (literature protocol)

**User skepticism (m2058)**: "honestly im still kinda skeptical that we reached 0.6 mPQ. while even LKCell is only 0.5". Audited the metric and re-ran evaluation against the **exact original dense PanNuke masks** (decoded from the fold3 parquet PNG bytes, the same target literature methods compare against) via `/tmp/opencode/orig_mask_eval.py`.

**Fold discovery**: the actual split is train=fold1 (2656), val=fold2 (2523), **test=fold3 (2722)** — the pannuke.yaml `split_map` (fold1=val, fold2=test, fold3=train) is stale. All test evals (train3/6/18-25) were already on fold3; the earlier fidelity measurement used fold2 (wrong fold).

**Results (train19 best.pt, fold3, conf=0.20, identical pred masks in both columns)**:

| metric | polygon-GT (current) | **original-mask GT** |
|---|---|---|
| bPQ (DQ/SQ) | 0.6728 (0.8333/0.8074) | **0.6901 (0.8318/0.8296)** |
| mPQ | 0.6294 | **0.6489** |
| AJI | 0.7521 | **0.7706** |
| centroid F1 (P/R) | 0.8823 (0.7937/0.9933) | **0.8749 (0.7913/0.9783)** |
| binary recall | 0.9381 (61,772/65,848) | 0.9301 (61,998/66,654) |

Per-class PQ (original-mask GT): Neoplastic 0.6916, Inflammatory 0.7195, Connective 0.6728, Necrosis 0.4610, Epithelial 0.6995.

**Interpretation**: evaluating against the original dense masks scores **higher** (bPQ 0.6901, mPQ 0.6489), not lower. My earlier 0.612/0.565 "fidelity-corrected estimate" double-counted boundary error multiplicatively and was wrong. The 64-ray GT polygon is a chord-inscribed inner approximation of the true boundary, so the model's slightly fuller polygons match the original shape better (SQ 0.830 vs 0.807). The original-mask column is also evaluated on a **harder** target: 66,654 instances vs 65,848 — the polygon ETL (`filter_and_clip_annotations`) drops 806 tiny nuclei across 645/2722 tiles, which are present in the original-mask GT.

**Vs literature (3-fold CV + watershed protocol)**: LSP-DETR bPQ 0.675/mPQ 0.482 (45M params), LKCell bPQ 0.684/mPQ 0.503 (163.8M). Our single-fold bPQ 0.6901/mPQ 0.6489 at 15.46M params beats both. Caveats: single fold (fold3 test), largest-first overlap resolution (vs watershed), no 3-fold CV averaging.

## Next
- train20 (pure-DETR, S/F/Q all off) → seed-arc completeness.
- Head-slimming ladder on this base: A (ndl 3→2), B (d_ffn 1024→512), E (A+B), C (hd 256→128), D (weight-shared layers).
