# Train36 — t12 + depthwise-separable decoder (deep sub-10M, ~7.85M)

## Run
- Date: 2026-09-05
- Branch: `feature/nulite-raycast-seg` (HEAD `9d70332`)
- Output: `output/nulite/train36` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs)
- Config: train35 recipe (`detr_hd: 128`, `detr_d_ffn: 1024`, `detr_shared_layers: true`, `detr_per_layer_match: true`, S+F+Q base, `detr_separable_decoder: true`) **+ `nulite_variant: fastvit_t12`** — the ONLY change from train35 (s12→t12). Model: **7.85M params**.

## Hypothesis
Combine the two sub-10M levers: t12 encoder (6.38M, full channels) + depthwise-separable decoder. Expect ~0.67-0.68 bPQ, i.e. the smallest model that still clears the literature bar.

## Results

### Val (train set; best.pt)
mAP50 0.4509, bPQ 0.5987, bSQ 0.7840, bDQ 0.7243 (fitness 0.5543). Slightly below train35 (0.6064) as expected for the smaller encoder.

### Test (fold3, 2722 tiles, conf=0.20, corrected metric)
AJI 0.7444 · AP@0.5 0.8669 · AP@0.7 0.6579 · AP@0.9 0.0191 · AP@0.5:0.05:0.95 0.4915 · **bPQ 0.6643** · **mPQ 0.6246** · F1 0.8728 · Prec 0.7836 · Rec 0.9849 · **7.85M** · 5.03 ms/img.

### Exact original-mask eval (literature protocol)
| metric | polygon-GT | **original-mask GT** |
|---|---|---|
| bPQ (DQ/SQ) | 0.6643 (0.8232/0.8070) | **0.6843 (0.8229/0.8315)** |
| mPQ | 0.6246 | **0.6493** |
| AJI | 0.7444 | **0.7647** |
| centroid F1 (P/R) | 0.8455 (0.7741/0.9433) | 0.8380 (0.7715/0.9291) |

Per-class PQ (orig GT): Neoplastic 0.6840, Inflammatory 0.7148, Connective 0.6629, Necrosis 0.4897, Epithelial 0.6953.

## Diagnosis
| run | config | bPQ | mPQ | orig bPQ/mPQ | params |
|---|---|---|---|---|---|
| train34 | t12 full-dec | 0.6808 | 0.6384 | 0.6993/0.6595 | 10.61M |
| train35 | s12 sep-dec | 0.6722 | 0.6301 | 0.6937/0.6536 | 9.77M |
| **train36** | **t12 sep-dec** | **0.6643** | **0.6246** | **0.6843/0.6493** | **7.85M** |

**t12 + separable decoder (7.85M):** orig-mask bPQ 0.6843 ties LKCell (0.684) and beats LSP-DETR (0.675); mPQ 0.6493 crushes both (0.503/0.482). This is the lightest model yet that clears the literature bar — 45M/163.8M competitors vs our 7.85M (~6×/21× smaller). Smallest Pareto point on the frontier.

## Next
- This completes the params/quality ladder: train31 (13.77M, 0.7138) → train34 (10.61M, 0.6993) → train35 (9.77M, 0.6937) → train36 (7.85M, 0.6843). Full Pareto front available for the paper.

---
