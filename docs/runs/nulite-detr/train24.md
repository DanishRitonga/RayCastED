# Train24 — decoder ablation (FastViT-direct DETR, no NuLite decoder / NP head)

## Run
- Date: 2026-08-30
- Branch: `feature/nulite-raycast-seg` (HEAD `b8d9877`)
- Output: `output/nulite/train24` (600 epochs, amp=false, patience=500, plain cosine, 0 zero-metric epochs)
- Config change from train23: `detr_use_decoder: true → false`, `lambda_seg: 1.0 → 0.0`, `detr_seed_feature: false` (unchanged), `detr_seed_in_content: false` (unchanged). The NuLite 5-block upsample decoder, `decoder0`, and the `NPHead` are **not constructed**; the DETR head reads the raw FastViT stage features directly (st4/st8/st16 = 64/128/256ch, exactly matching `ch=(64,128,256)`). `detr_mds: false`, `detr_cost_inside: 10.0`, `detr_no_object_weight: 2.0`, grid queries, 3 layers hd=256 unchanged.

## Hypothesis
Answer "is the decoder (and thus the NP head) even needed?" The encoder already emits native multi-scale features at the exact channels the head consumes; the decoder exists solely to produce b3/b4/b5 (which the encoder has natively) plus the stride-1 b1 feeding the NP head. If FastViT-direct ≈ train20 pure-DETR (bPQ 0.6565, which still had decoder features in the forward path) → the whole decoder + NP subsystem is strippable (~3.26M params saved, model becomes FastViT→DETR at ~12.2M).

## Results

### Val (train set; still climbing at ep600)
Best fitness ep599 (0.5191): bPQ 0.5619, bSQ 0.7777, bDQ 0.6858, mAP50 0.4193, mAP50-95 0.3094, P 0.4262, R 0.6563.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
| metric | train19 S+F+Q | train20 none | train23 S | train24 no-decoder |
|---|---|---|---|---|
| bPQ | **0.6728** | 0.6565 | 0.6652 | 0.6355 |
| mPQ | **0.6294** | 0.6160 | 0.6234 | 0.6104 |
| AP@0.5 | **0.8758** | 0.8682 | 0.8728 | 0.8675 |
| AP@0.5:0.95 | **0.5023** | 0.4803 | 0.4874 | 0.4899 |
| AJI | 0.7521 | 0.7388 | 0.7464 | 0.7327 |
| F1 | 0.8823 | 0.8743 | 0.8813 | 0.8417 |
| Prec / Rec | 0.7937 / 0.9933 | 0.7812 / 0.9925 | 0.7925 / 0.9925 | 0.7308 / 0.9923 |
| preds | 82,409 (1.25x) | 83,658 (1.27x) | 82,462 (1.25x) | 89,417 (1.36x) |
| ms/img | 5.10 | 5.28 | 5.09 | **4.47** |
| params | 15.46M | 15.46M | 15.46M | **12.21M** |

GFLOPs printed as 5.52 for all runs — this is a **hardcoded placeholder** in eval_pannuke.py (line 977-980: `5.52 * width_multiple²`), NOT measured; ms/img is the real latency. Train24 params 12.21M (8.30M encoder + 3.91M head). Class centroid F1: Neo 0.8138 / Inflamm 0.8985 / Conn 0.8359 / Necr 0.7894 / Epi 0.8460. Size recall: small 0.9799 / medium 0.9975 / large 0.9994. Matched 65,344/65,848 GT. Tissue macro: bPQ 0.6487 ± 0.0470, mPQ 0.6845 ± 0.0456, AJI 0.7475 ± 0.0479 (worst Colorectal bPQ 0.5178). Polygon-level (area-ratio metric, relative only): bPQ 0.8026, centroid L2 88.86px.

## Diagnosis
**The decoder is needed.** FastViT-direct (train24) is the weakest config in the arc, well below the decoder-feature baselines:
- vs train20 (identical pure-DETR but *with* decoder features): **−0.021 bPQ, −3.3 F1, −5.0 precision, +0.09x preds**. Recall identical (0.9923).
- vs train19 (full seed): −0.037 bPQ, −4.1 F1, −6.3 precision.

Interpretation: AP@0.5/50-95 are ~flat (0.8675/0.4899 vs 0.8682/0.4803) — the decoder does **not** help the model *find* nuclei. It helps **winner-selection**: raw st4/st8/st16 features produce more duplicate/FP detections (1.36x vs 1.27x) and lower precision. The decoder's upsampled b3/b4/b5 integrate coarse-to-fine context that the DETR head's deformable cross-attention needs to pick the single winning query per nucleus — consistent with the earlier ConvTranspose smear finding that the decoder's stride-4 output is smoother than the encoder's raw st4 (the smoothness is what the head wants for cross-scale matching).

The full architecture question is settled: **keep the NuLite decoder + NP head**. train19 (S+F+Q) remains the best config and the head-ladder base.

## Side benefit
Train24 is a clean lightweight-variant data point for the paper's scaling table: **12.21M params, 4.47 ms/img, bPQ 0.636 / mPQ 0.610** — competitive quality at ~21% fewer params and ~12% faster than the full model (vs LSP-DETR 45M / 26G). FastViT-direct is the "small" config if the paper wants a size/latency ladder.

## Next
- train25 (S+Q): complete the seed-arc matrix (missing cell). `detr_use_decoder: true`, `lambda_seg: 1.0`, `detr_seed_feature: false`, `detr_seed_in_content: true`. Determines whether the feature channel (F) is redundant when the objectness read (Q) is present → decides the head-ladder base (S+Q vs S+F+Q).
- Then head-slimming ladder on the winner base: A (detr_ndl 3→2), B (detr_d_ffn 1024→512, needs threading), E (A+B), C (detr_hd 256→128), D (weight-shared layers).
