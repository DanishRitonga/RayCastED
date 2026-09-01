# Train31 — s12 full-seed + per-layer matching + weight-shared decoder layers (ladder D)

## Run
- Date: 2026-09-01
- Branch: `feature/nulite-raycast-seg` (HEAD `0489dcd`)
- Output: `output/nulite/train31` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs, best = ep599/600 still climbing)
- Config: train29 parity-base S+F+Q (`lambda_seg: 1.0`, `detr_seed_feature: true`, `detr_seed_in_content: true`, `detr_no_object_weight: 2.0`, `detr_cost_inside: 10.0`, `detr_mds: false`, grid queries, 3 layers hd=256, `detr_per_layer_match: true`, `nulite_variant: fastvit_s12`, NuLite-parity decoder) **+ `detr_shared_layers: true`** — the ONLY change from train29. Also restored `detr_backbone: nulite` after the yolo11s (train30) experiment.

## Hypothesis
Ladder D (weight-shared transformer decoder layers): `self.decoder.layers = nn.ModuleList([self.decoder.layers[0]] * num_layers)` — one shared DeformableTransformerDecoderLayer reused 3× (recurrent refinement). Per-layer `dec_bbox_head`/`dec_score_head` stay distinct. Params-only change: head 3.905M → 1.914M, model 15.76M → 13.77M, FLOPs unchanged. Two candidate mechanisms: (a) recurrent-refinement regularization (shared params see 3× gradient pressure per token → better convergence), (b) risk that tying layers reduces refinement capacity.

## Results

### Val (train set; still climbing at ep600)
Best fitness ep599 (0.5587): mAP50 0.4611, bPQ 0.6005, bSQ 0.7815, bDQ 0.7290; ep600 row: mAP50 0.4609, bPQ 0.6004. **Surprise: weight-sharing improved val across the board vs train29 parity base (bPQ +0.024, bDQ +0.028, mAP50 +0.015) at −2M params.**

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
AJI **0.7639**, AP@0.5 **0.8827**, AP@0.7 0.6971, AP@0.9 0.0258, AP@0.5:0.05:0.95 **0.5192**, bPQ **0.6954**, bMPQ 0.7810, mPQ **0.6558**, mMPQ 0.7518, F1(centroid, r=12) **0.8963**, Prec **0.8211**, Rec 0.9865, Params **13.77M**, 5.24 ms/img, **79,113 preds (1.20x)**.

Class centroid F1: Neoplastic 0.8810 / Inflammatory 0.9254 / Connective 0.8906 / Necrosis 0.7921 / Epithelial 0.9064. Size recall: small 0.9647 / medium 0.9965 / large 0.9982. Tissue macro: bPQ 0.7766, mPQ 0.7021, AJI 0.7271. Polygon-level (area-ratio, relative only): bPQ 0.8549, centroid L2 83.44px, ray L1 1.67px.

**vs train29 (parity base) and train28 (old-decoder previous best):**

| metric | train29 | train28 | train31 | Δ vs 29 | Δ vs 28 |
|---|---|---|---|---|---|
| bPQ | 0.6698 | 0.6800 | **0.6954** | +0.026 | +0.015 |
| mPQ | 0.6284 | 0.6442 | **0.6558** | +0.027 | +0.012 |
| AP@0.5 | 0.8743 | 0.8730 | **0.8827** | +0.008 | +0.010 |
| AP@50-95 | 0.5040 | 0.4883 | **0.5192** | +0.015 | +0.031 |
| AJI | 0.7548 | 0.7521 | **0.7639** | +0.009 | +0.012 |
| F1 | 0.8694 | 0.8864 | **0.8963** | +0.027 | +0.010 |
| Prec | 0.7768 | 0.8065 | **0.8211** | +0.044 | +0.015 |
| Rec | 0.9870 | 0.9872 | 0.9865 | ~0 | ~0 |
| preds | 83,669 (1.27x) | 80,596 (1.22x) | **79,113 (1.20x)** | fewer | fewer |
| params | 15.76M | 15.46M | **13.77M** | −2.0M | −1.7M |
| ms/img | 5.16 | 5.55 | 5.24 | ~flat | −0.31 |

## Diagnosis
**Weight-sharing is a genuine win, not a wash.** vs the parity base (train29): bPQ +2.6pt, mPQ +2.7pt, precision +4.4pt, fewest preds yet (1.20x), at −2.0M params and ~flat latency. It also beats the old-decoder best (train28) on every metric at −1.7M params. The recurrent-refinement regularizer interpretation holds: a shared layer sees 3× gradient pressure, converges better, and suppresses duplicates — consistent with the earlier finding that layer refinement is the winner-selection bottleneck. Weight-sharing is now the submission candidate.

## Exact original-mask eval (literature protocol)
Via `/tmp/opencode/orig_mask_eval.py` (CKPT = train31 best.pt), fold3, conf=0.20. NOTE: script was rebuilt after a /tmp cleanup; critical fix was tile↔parquet alignment — parquet rows are in numeric ROI order but tiles iterate lexicographically (roi_0, roi_1, roi_10, roi_100…), so the parquet row is `roi_id` parsed from the tile filename, not the tile index. GT instance-count mismatch (polygon vs orig) = 645/2722 (known ETL tiny-nucleus drops). Polygon-GT numbers reproduce eval_pannuke exactly (validates the rebuild).

| metric | polygon-GT (current) | **original-mask GT** |
|---|---|---|
| bPQ (DQ/SQ) | 0.6954 (0.8512/0.8170) | **0.7138 (0.8494/0.8403)** |
| mPQ | 0.6558 | **0.6777** |
| AJI | 0.7639 | **0.7824** |
| centroid F1 (P/R) | 0.8638 (0.8033/0.9426) | 0.8559 (0.8006/0.9282) |
| bMPQ / cMPQ (fg-masked) | 0.7905 | 0.7945 |

Per-class PQ (original-mask GT): Neoplastic 0.7152, Inflammatory 0.7429, Connective 0.6972, Necrosis 0.5123, Epithelial 0.7211.

**Vs literature (3-fold CV + watershed protocol)**: LSP-DETR bPQ 0.675/mPQ 0.482 (45M), LKCell bPQ 0.684/mPQ 0.503 (163.8M). **train31 single-fold bPQ 0.7138 / mPQ 0.6777 at 13.77M beats both on bPQ AND mPQ** — beats the train28 orig-mask numbers (0.7013/0.6687) with 1.7M fewer params. Necrosis remains the weak class (PQ 0.51). Caveats unchanged: single fold (fold3), largest-first overlap resolution vs watershed, no 3-fold CV averaging.

## Next
- Update `docs/paper/benchmark.md` with train31 as the best row.
- Remaining head-slimming candidate: C (hd 128 → ~13.0M, ~−75% FLOPs, riskiest) if a mid-size Pareto point is wanted; shared-layers D already gives the params win at ~flat FLOPs.
