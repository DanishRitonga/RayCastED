# Train33 — ladder C: decoder hidden dim hd 256 → 128 (shared-layers base)

## Run
- Date: 2026-09-01
- Branch: `feature/nulite-raycast-seg` (HEAD `ec42b02`)
- Output: `output/nulite/train33` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs, best = ep600 still climbing)
- Config: train31 shared-layers base S+F+Q (`lambda_seg: 1.0`, `detr_seed_feature: true`, `detr_seed_in_content: true`, `detr_no_object_weight: 2.0`, `detr_cost_inside: 10.0`, `detr_mds: false`, `detr_per_layer_match: true`, `detr_shared_layers: true`, `detr_d_ffn: 1024`, `nulite_variant: fastvit_s12`) **+ `detr_hd: 256→128`** — the ONLY change from train31. Model: **12.53M params** (−1.24M vs train31). The last head-slimming knob (riskiest: ~−75% transformer FLOPs).

## Hypothesis
Ladder C cuts the whole hidden dimension (QKV projections + all FFN widths shrink). Prior cuts (ndl→2, d_ffn→512) regressed, but hd=128 pushes into NuLite-sized territory (bare NuLite s12 = 12.04M): a full query-based polygon DETR head packed into +0.5M over NuLite's seg heads, NMS/post-proc-free.

## Results

### Val (train set; best ep600, still climbing)
Best fitness ep600 (0.5639): mAP50 0.4617, bPQ 0.6077, bSQ 0.7817, bDQ 0.7366. **Surprise: hd=128 val beat hd=256 (train31: bPQ +0.007, bDQ +0.008)** — narrower hidden dim acts as a regularizer like shared-layers.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
AJI 0.7414 · AP@0.5 0.8789 · AP@0.7 0.6379 · AP@0.9 0.0150 · AP@0.5:0.05:0.95 0.4808 · **bPQ 0.6755** · **mPQ 0.6349** · F1 0.8965 · Prec 0.8219 · Rec 0.9860 · **12.53M** · **5.05 ms/img** · 78,991 preds (1.20x). Class F1: Neo 0.8838/Inflamm 0.9206/Conn 0.8893/Necr 0.7883/Epi 0.9038. Tissue macro bPQ 0.6821.

**vs train31 (hd=256):** bPQ −0.020, mPQ −0.021, AP@50-95 −0.038, AJI −0.022, ms −0.19, params −1.24M; F1/Prec/Rec ~equal. The big val win did NOT hold on test (same val/test divergence as train28/29).

### Exact original-mask eval (literature protocol, via /tmp/opencode/orig_mask_eval.py)
| metric | polygon-GT | **original-mask GT** |
|---|---|---|
| bPQ (DQ/SQ) | 0.6755 (0.8472/0.7973) | **0.6954 (0.8463/0.8217)** |
| mPQ | 0.6349 | **0.6580** |
| AJI | 0.7414 | **0.7618** |
| centroid F1 (P/R) | 0.8640 (0.8037/0.9433) | 0.8560 (0.8009/0.9287) |
| bMPQ / cMPQ | 0.7792 | 0.7841 |

Per-class PQ (orig GT): Neoplastic 0.7003, Inflammatory 0.7145, Connective 0.6792, Necrosis 0.4920, Epithelial 0.7038.
**Vs literature (3-fold CV + watershed):** LSP-DETR bPQ 0.675/mPQ 0.482 (45M) → beaten on both (0.6954/0.6580); LKCell 0.684/0.503 (163.8M) → beaten on both; NuLite ≈0.68 bPQ → marginally above at essentially NuLite's params.

## Diagnosis
hd=128 is a **usable Pareto point**: −1.8pt bPQ vs hd=256 (train31) for −1.24M params / −0.19ms, still beating all literature. Nearest-to-NuLite params (12.53M vs 12.04M) with full end-to-end polygon detection → the strongest "convert a post-proc model into a lighter end-to-end one" narrative. Adopted as the submission config (user decision).

## Ladder summary (all on shared-layers base unless noted)
| ladder | change | test bPQ | verdict |
|---|---|---|---|
| A train26 | ndl 3→2 | 0.647 | regress — revert |
| B train32 | d_ffn 1024→512 | 0.684 | small regress — revert |
| C train33 | hd 256→128 | 0.676 | usable Pareto point (12.53M) — ADOPTED |
| D train31 | shared layers | **0.6954** | win — high-quality alternative |

---
