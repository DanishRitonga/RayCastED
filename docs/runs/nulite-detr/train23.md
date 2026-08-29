# Train23 — seg-loss-only ablation (S, F/Q OFF)

## Run
- Date: 2026-08-30
- Branch: `feature/nulite-raycast-seg` (HEAD `6adc532`)
- Output: `output/nulite/train23` (600 epochs, amp=false, patience=500, plain cosine, 0 zero-metric epochs)
- Config change from train22: `detr_seed_feature: true → false`, `detr_seed_in_content: false` (unchanged) → **S only** (seg loss on the InstanSeg seed-map target, `lambda_seg: 1.0`, `seed_map_target: true`), seed feature channel + seed_q read OFF. `detr_mds: false`, `detr_cost_inside: 10.0`, `detr_no_object_weight: 2.0`, grid queries, 3 layers hd=256 unchanged.

## Hypothesis
Isolate the seg loss alone. If S-only ≈ 0.66 → the seg loss is the main contributor and F/Q are marginal on top. If S-only ≈ train20 (0.657) → the loss alone is inert and the seed gain requires F/Q jointly. Decision rule from train22: "decisive for whether the seg loss alone is inert (→ F+Q jointly required) or ≈ 0.66 (→ F/Q marginal)".

## Results

### Val (train set; still climbing at ep600)
Best fitness ep600 (0.5495): bPQ 0.5904, bSQ 0.7802, bDQ 0.7185, mAP50 0.4539, mAP50-95 0.3452, P 0.4515, R 0.6565.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
| metric | train19 S+F+Q | train20 none | train21 Q | train22 S+F | train23 S |
|---|---|---|---|---|---|
| bPQ | **0.6728** | 0.6565 | 0.6474 | 0.6598 | 0.6652 |
| mPQ | **0.6294** | 0.6160 | 0.6046 | 0.6186 | 0.6234 |
| AP@0.5 | **0.8758** | 0.8682 | 0.8495 | 0.8697 | 0.8728 |
| AP@0.5:0.95 | **0.5023** | 0.4803 | 0.4883 | 0.4755 | 0.4874 |
| AJI | 0.7521 | 0.7388 | 0.7426 | 0.7413 | 0.7464 |
| F1 | 0.8823 | 0.8743 | 0.8540 | 0.8793 | 0.8813 |
| Prec / Rec | 0.7937 / 0.9933 | 0.7812 / 0.9925 | 0.7502 / 0.9911 | 0.7893 / 0.9926 | 0.7925 / 0.9925 |
| preds | 82,409 (1.25x) | 83,658 (1.27x) | 86,992 (1.32x) | 82,809 (1.26x) | 82,462 (1.25x) |
| ms/img | 5.10 | 5.28 | 5.19 | 5.11 | 5.09 |

Params 15.46M / GFLOPs 5.52. Class centroid F1: Neo 0.8606 / Inflamm 0.9103 / Conn 0.8721 / Necr 0.7605 / Epi 0.8910. Size recall: small 0.9808 / medium 0.9978 / large 0.9988. Matched 65,355/65,848 GT. Polygon-level (area-ratio metric, relative only): bPQ 0.8392.

## Diagnosis
**S-only is the second-best config in the arc** (bPQ 0.6652), +0.9 bPQ over pure-DETR (train20). Component ranking (test bPQ):
`S+F+Q (0.6728) > S (0.6652) > S+F (0.6598) > none (0.6565) > Q (0.6474)`.
The seg loss is the **main contributor** of the seed branch; the F channel alone *hurts* (−0.5 on top of S), and Q alone is harmful but unlocks the feature channel's value in the full combo. The nonlinear interaction (train22) is confirmed and now precisely attributed: **S carries the gain; F needs Q to be useful; Q needs S+F to be useful**. Full seed (train19) stays the best config and the head-ladder base.

## Next
- train24 (decoder ablation): `detr_use_decoder: false` → bypass the NuLite decoder + NP head entirely; DETR head reads raw FastViT stage features (strides 4/8/16, 64/128/256ch). `lambda_seg: 0.0`, seed_feature/seed_in_content false. ~12.2M params. Decision: ≈ train20 (0.657) → decoder redundant → FastViT→DETR direct; ≥ 0.673 → decoder actively hurts; well below 0.657 → decoder features carry value.
