# Train35 — s12 + depthwise-separable decoder (sub-10M)

## Run
- Date: 2026-09-05
- Branch: `feature/nulite-raycast-seg` (HEAD `88e47c9`)
- Output: `output/nulite/train35` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs)
- Config: train33/train34 submission recipe (`detr_hd: 128`, `detr_d_ffn: 1024`, `detr_shared_layers: true`, `detr_per_layer_match: true`, S+F+Q base, patience 300, amp false) **+ `detr_separable_decoder: true`, `nulite_variant: fastvit_s12`**. Model: **9.77M params**.

## Hypothesis
Thin the NuLite decoder by factorizing its 3×3 convs into depthwise-separable (3×3 depthwise → 1×1 pointwise) to cross sub-10M. The decoder's two 512→256 projections alone are 2.36M of the 3.51M decoder; separable factorization cuts decoder convs ~3.14M → ~0.365M. Keeps 3×3 per-channel spatial mixing (lower risk than 1×1-only). Expect quality to hold within ~1pt bPQ while landing under 10M.

## Results

### Val (train set; best.pt)
mAP50 0.4566, bPQ 0.6064, bSQ 0.7838, bDQ 0.7346, P 0.4671, R 0.6464 (fitness 0.5614). Essentially flat vs full-decoder val (train33 0.6077 / train34 0.6095).

### Test (fold3, 2722 tiles, conf=0.20, corrected metric)
AJI 0.7462 · AP@0.5 0.8704 · AP@0.7 0.6582 · AP@0.9 0.0173 · AP@0.5:0.05:0.95 0.4902 · **bPQ 0.6722** · **mPQ 0.6301** · F1 0.8857 · Prec 0.8031 · Rec 0.9872 · **9.77M** · 5.27 ms/img.

### Exact original-mask eval (literature protocol)
| metric | polygon-GT | **original-mask GT** |
|---|---|---|
| bPQ (DQ/SQ) | 0.6722 (0.8353/0.8048) | **0.6937 (0.8347/0.8311)** |
| mPQ | 0.6301 | **0.6536** |
| AJI | 0.7462 | **0.7687** |
| centroid F1 (P/R) | 0.8586 (0.7926/0.9470) | 0.8506 (0.7897/0.9323) |

Per-class PQ (orig GT): Neoplastic 0.6946, Inflammatory 0.7201, Connective 0.6759, Necrosis 0.4716, Epithelial 0.7057.

## Diagnosis
| run | backbone/dec | bPQ | mPQ | params |
|---|---|---|---|---|
| train34 | t12 full | 0.6808 | 0.6384 | 10.61M |
| train33 | s12 full | 0.6755 | 0.6349 | 12.53M |
| **train35** | **s12 sep** | **0.6722** | **0.6301** | **9.77M** |

**Separable decoder = small regression (−0.006 to −0.009 bPQ, −0.005 to −0.008 mPQ) for −2.76M params.** Sub-10M achieved (9.77M) at a slight quality cost — a clean Pareto point. Orig-mask bPQ 0.6937 still beats LSP-DETR (0.675) and LKCell (0.684); mPQ 0.6536 far exceeds both (0.482/0.503).

## Next
- train36 (t12 + separable decoder, ~7.85M) — the deeper sub-10M point.

---
