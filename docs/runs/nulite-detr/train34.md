# Train34 — FastViT t12 backbone (sub-10M exploration)

## Run
- Date: 2026-09-01
- Branch: `feature/nulite-raycast-seg` (HEAD `efd404a`)
- Output: `output/nulite/train34` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs, best = ep590/600 still climbing)
- Config: train33 submission recipe (`detr_hd: 128`, `detr_d_ffn: 1024`, `detr_shared_layers: true`, `detr_per_layer_match: true`, S+F+Q base: `lambda_seg: 1.0`/`detr_seed_feature: true`/`detr_seed_in_content: true`/`detr_no_object_weight: 2.0`/`detr_cost_inside: 10.0`/`detr_mds: false`, ndl 3, grid 0.05→400 q, patience 300, amp false) **+ `nulite_variant: fastvit_t12`** — the ONLY change from train33. Model: **10.61M params** (encoder 6.38M + decoder 3.51M + detr head 0.68M).

## Hypothesis
FastViT t12 has the SAME embed_dims [64,128,256,512] as s12 but a 6.38M encoder (vs s12's 8.30M). Since the NuLite decoder + DETR head are channel-derived, they are byte-identical to s12 — only the encoder differs. Prediction: quality tracks s12 within noise at −1.9M params. This is the greedy "sub-10M-adjacent" path (t8 was true sub-10M at 8.89M but lost −6pt bPQ via half-width channels [48,96,192,384]).

## Results

### Val (train set; best ep590, still climbing)
mAP50 0.4577, bPQ 0.6095, bSQ 0.7813, bDQ 0.7398 (fitness 0.5639). ≈ s12 train33 val (0.4617/0.6077/0.7817/0.7366).

### Test (fold3, 2722 tiles, conf=0.20, corrected metric)
AJI 0.7470 · AP@0.5 0.8737 · AP@0.7 0.6539 · AP@0.9 0.0168 · AP@0.5:0.05:0.95 0.4887 · **bPQ 0.6808** · **mPQ 0.6384** · F1 0.8946 · Prec 0.8201 · Rec 0.9839 · **10.61M** · 5.11 ms/img.

**vs train33 (s12 hd128): bPQ +0.005 (0.6808 vs 0.6755), mPQ +0.004 (0.6384 vs 0.6349) at −1.9M params.** Full-channel prediction HELD — quality tracks within noise (even slightly better) at 10.61M.

### Exact original-mask eval (literature protocol, via /tmp/opencode/orig_mask_eval.py)
| metric | polygon-GT | **original-mask GT** |
|---|---|---|
| bPQ (DQ/SQ) | 0.6808 (0.8463/0.8044) | **0.6993 (0.8449/0.8276)** |
| mPQ | 0.6384 | **0.6595** |
| AJI | 0.7470 | **0.7667** |
| centroid F1 (P/R) | 0.8652 (0.8060/0.9426) | 0.8565 (0.8026/0.9274) |
| bMPQ / cMPQ | 0.7805 | 0.7848 |

Per-class PQ (orig GT): Neoplastic 0.7039, Inflammatory 0.7225, Connective 0.6795, Necrosis 0.4813, Epithelial 0.7105.
**Vs literature (3-fold CV + watershed):** LSP-DETR bPQ 0.675/mPQ 0.482 (45M) → beaten on both (0.6993/0.6595); LKCell 0.684/0.503 (163.8M) → beaten on both; NuLite ≈0.68 bPQ → above at 10.61M (BELOW bare NuLite's 12.04M).

## Diagnosis
**t12 is the greedy win.** At 10.61M it matches (slightly beats) s12 quality on both protocols and beats all literature, while being smaller than bare NuLite (12.04M). The −1.9M params come purely from the encoder, with zero quality cost. t12 at 10.61M is the best params↔quality point yet; if a strict <10M budget is required, decoder thinning (width-scale the 3.51M decoder) is the remaining lever.

## Next
- Optional: decoder-width thinning to cross <10M (path B).
- Otherwise t12 (train34) joins train33 (hd128 s12) as a Pareto front; train31 (hd256 s12, 13.77M, orig-mask 0.7138) remains the absolute-quality ceiling.

---
