# Train22 — seg-loss + feature-channel ablation (seed_q OFF)

## Run
- Date: 2026-08-29
- Branch: `feature/nulite-raycast-seg` (HEAD `c123d6e`)
- Output: `output/nulite/train22` (600 epochs, amp=false, patience=500, plain cosine)
- Config change from train21: `lambda_seg: 0.0 → 1.0`, `detr_seed_feature: false → true`, `detr_seed_in_content: true → false` → S+F (seg loss + seed-map feature channel), seed_q OFF. `seed_map_target: true` (center-weighted target), `detr_mds: false`, `detr_cost_inside: 10.0`, `detr_no_object_weight: 2.0` unchanged.
- Launched config verified from live checkpoint (ep432, model attrs): `seed_map_target=True`, `lambda_seg=1.0`, `cost_inside=10.0`, `no_object_weight=2.0`, `query_selection=grid`, `input_proj` channels 65/129/257 (seed_feature on, +1 channel/scale), 3 decoder layers hd=256.

## Hypothesis
Isolate the seed_q objectness read: if S+F ≈ train19 (0.673), seed_q is pure harm and S+F is the minimal seed recipe. If S+F ≈ train20 (pure DETR), the seed gain needs seed_q and the minimal recipe is full seed.

## Results

### Val (train set; 0 zero-metric epochs)
Best fitness ep599: bPQ 0.5907, bSQ 0.7808, bDQ 0.7172, mAP50 0.4530, P 0.4512, R 0.6599. Best bPQ ep600 0.5909. Still climbing at ep600.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
| metric | train19 S+F+Q | train20 none | train22 S+F |
|---|---|---|---|
| bPQ | **0.6728** | 0.6565 | 0.6598 |
| mPQ | **0.6294** | 0.6160 | 0.6186 |
| AP@0.5 | **0.8758** | 0.8682 | 0.8697 |
| AP@0.5:0.95 | **0.5023** | 0.4803 | 0.4755 |
| AJI | 0.7521 | 0.7388 | 0.7413 |
| F1 | 0.8823 | 0.8743 | 0.8793 |
| Prec / Rec | 0.7937 / 0.9933 | 0.7812 / 0.9925 | 0.7893 / 0.9926 |
| preds | 82,409 (1.25x) | 83,658 (1.27x) | 82,809 (1.26x) |
| ms/img | 5.10 | 5.28 | 5.11 |

Params 15.46M / GFLOPs 5.52. Class centroid F1: Neo 0.8606 / Inflamm 0.9091 / Conn 0.8722 / Necr 0.7594 / Epi 0.8910. Size recall: small 0.9812 / medium 0.9975 / large 0.9989. Matched 65,359/65,848 GT; 489 unmatched, 71.2% truly missed (>12px). Polygon-level (area-ratio metric, relative only): bPQ 0.8374, bSQ 0.9540, bDQ 0.8778, centroid L2 86.09px, ray L1 1.70px.

## Diagnosis
**S+F without seed_q is a marginal wash** — test bPQ 0.6598 ≈ train20 (pure DETR, 0.6565, Δ0.003 = noise). Dropping seed_q erases essentially the entire seed-branch gain. Combined with train21 (Q alone 0.6474 < none), the components form a **nonlinear interaction**: Q is inert/harmful until S has trained the NP head and F carries its structure into cross-attention; then the per-query center/objectness prior is worth +0.013 bPQ / +0.027 AP@0.5:0.95 over S+F. The full seed branch (S+F+Q, train19) remains the best config.

## Next
- train23 (S only): `lambda_seg: 1.0`, `detr_seed_feature: false`, `detr_seed_in_content: false`, `seed_map_target: true`. Decisive for whether the seg loss alone is inert (→ F+Q jointly required) or ≈ 0.66 (→ F/Q marginal).
- Head-ablation ladder base = train19 full-seed config (S+F+Q).
