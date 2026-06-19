# AGENTS.md - RayCastED Instructions

## PRIME DIRECTIVE: NO POST-PROCESSING

**The entire point of this project is NMS-free, post-processing-free inference.** The model must produce the correct set of predictions directly from the o2o head — no NMS, no deduplication, no clustering, no confidence thresholding tricks, no post-hoc filtering. If the model overpredicts, the fix must be in **training** (loss, assignment, cls signal quality), never in post-processing.

This means:
- topk2=1 at inference is NOT a post-processing step — it's the training assignment ensuring 1:1 matching
- `postprocess()` in head.py only does top-k selection by confidence (inherent to E2E design), NOT deduplication
- DO NOT add NMS, DO NOT add anchor clustering, DO NOT add spatial deduplication
- Overprediction must be solved by making the o2o cls head confident enough to suppress false positives

## Core Commands

```bash
# Environment setup
uv sync

# Code quality
uv run ruff check .
uv run ruff format .

# Testing (all tests)
uv run python -m pytest tests/phase_6/ -x -q

# Cleaning (cache reset)
bash clean_cache.sh
```

**NEVER launch training on this device.** Training runs happen on a separate GPU machine. Only code changes, config edits, testing, and linting are done locally.

## Architecture & Data Flow

**Main directories:**
- `raycasted/model/` — model definitions, train.py, loss.py, tal.py
- `main/pannuke.yaml` — training config
- `tests/phase_6/` — 28 pytest tests

**Key files by role:**
- `raycasted/model/blocks/head.py` — RayCastDetect head (separate o2o heads), bias_init, postprocess
- `raycasted/model/loss.py` — RayCastDetectionLoss + RayCastE2ELoss (dual assignment)
- `raycasted/model/tal.py` — RayCastAssigner (greedy TAL) + HungarianRayCastAssigner (unused)
- `raycasted/data/etl/utils/constants.py` — N_RAYS, ANGLE, RAY_COS/SIN state (module-level, late-binding)
- `raycasted/scripts/eval_pannuke.py` — streaming eval (AJI, bPQ, mPQ, AP, F1)

**Execution flow:**
```
train.py → get_model() (nc=self.data['nc'], end2end=True)
  ↓
RayCastE2ELoss → one2many (RayCastDetectionLoss, topk=15) + one2one (topk=7, topk2=1)
  ↓
Both branches use RayCastAssigner (NOT Hungarian — HungarianRayCastAssigner exists but is unwired)
  ↓
Inference: fuse() → keeps only o2o heads (one2one_cv2 + one2one_cv3) → postprocess(max_det=300)
```

**Architecture:** P2-P3-P4 pyramid (strides 4/8/16), 5376 total anchors at 256px input.

**IMPORTANT: n_rays=64 in ALL training runs.** While `constants.py` defaults to 32 rays, `main/pannuke.yaml` sets `n_rays: 64` and this has been the case for every experiment (including the logged results below). The model head, loss functions, and assigner all use `n_rays` from config. Always check `pannuke.yaml` for the active n_rays value.

## Training Config Threading Pattern

When adding a new config parameter:
1. Add to `_RayCastCriterionWrapper.__call__` config parsing (train.py:~180)
2. Pass to `RayCastE2ELoss.__init__` (loss.py:~647)
3. Pass to `RayCastDetectionLoss.__init__` (loss.py:~280)
4. Pass to assigner constructors in loss.py
5. Add to `TrainingConfig` in `raycasted/data/etl/utils/config.py`
6. Add to main/pannuke.yaml
7. Run: `ruff check && ruff format && pytest tests/phase_6/ -x -q`

## Critical Gotchas

### Classification & Overprediction

1. **bg_cls_decay applied once (verified)**: `loss.py:524-537` applies bg_cls_decay once (or zero times when `bg_cls_decay=1.0`, which is the current default — the guard skips entirely). No double-application exists.

2. **focal_alpha=0.5 causes mode collapse**: With alpha=0.5, focal loss gives equal weight to fg and bg. Combined with `bg_fg_ratio=3` (3x more bg anchors sampled), the effective bg:fg gradient ratio is 3:1. This overwhelms the 5 real classes. Use `focal_alpha=0.75`.

3. **class_weights scales the loss, not the assignment**: `class_weights` in loss.py scales positive cls loss magnitudes from the first epoch — there is no warmup gate. It is disabled because it caused mode collapse (fg-only amplification with weak bg gradient pushed model to predict the least-penalized class).

4. **Shared cv3 fix (verified)**: O2O branch has separate `one2one_cv3` cls head via `copy.deepcopy`. `fuse()` sets `cv2=cv3=None`, keeping only o2o heads. Shared head caused 11.5x overprediction.

5. **Separate head alone doesn't fix overprediction**: Even with separate o2o cls head, model still produces 11.3x overprediction (746k preds vs 66k GT). The o2o cls head gets too few positives (topk2=1 → 28 fg/image vs o2m's 420) to learn good fg/bg discrimination. Overprediction must be fixed via training signal, NOT post-processing.

6. **Hard binarization destroys quality signal**: `loss.py:480` sets `cls_targets[cls_targets > 0] = 1.0` (on a cloned tensor, so no autograd issue). All positives get same target regardless of match quality. Soft targets for o2o may help.

42. **Detection is the bottleneck, not segmentation**: bSQ reaches 0.756 by epoch 5 (when it finds a nucleus, it draws good boundaries), but bDQ=0.322 (it can't find them reliably). bPQ = DQ × SQ — DQ is always the limiting factor. All regression-targeting approaches (DCN, quality head, bound_l1, range_l1) are dead ends because the problem is cls discrimination, not boundary quality.

### Assignment & Warmup

7. **warmup sigma must be configurable**: Sigma was previously hardcoded at 0.15. Now configurable via `assigner_radius_scale` in pannuke.yaml. Always thread sigma through loss.py → tal.py and update pannuke.yaml.

8. **assigner_radius_scale must match model geometry**: Default fallback [8,16,32] in tal.py is wrong for P2-P3-P4 architecture. In practice the model's actual strides are always passed, so the fallback is never used. Latent risk only if `RayCastAssigner()` is called directly without stride.

9. **dynamic topk masks garbage**: `select_topk_candidates` in tal.py masks out zero-metric entries. You can increase tal_topk to get more candidates through the containment filter.

10. **2-phase Hungarian blending**: Hungarian assigner for o2o blended via smooth schedule in `RayCastE2ELoss`. Phase 1 (0→p2): pure TAL. Phase 2 (p2→end): Hungarian ramps 0→max_weight, TAL fades as `1 - hw`. The old 3-phase system is deprecated — `hungarian_phase3_start` is ignored. Config: `hungarian_phase2_start` (use -1 for auto), `hungarian_max_weight`, `hungarian_cost_{class,centroid,ray}`. Implementation: `_compute_hungarian_o2o_loss()` temporarily swaps the o2o assigner.

11. **Topk2 annealing**: `o2o_topk2_start` (default 3) and `o2o_topk2_anneal_epoch` (default 120) linearly anneal topk2 from start→1 over [anneal_epoch, max_epochs]. Inspired by One-to-Few (CVPR 2023). Config in pannuke.yaml.

### Model Construction

12. **nc must be set at construction time**: Pass `nc=self.data['nc']` in `get_model()` (train.py). If nc is set after construction, cv3 conv layers stay at 80 channels.

13. **bias_init default (256) correct for PanNuke**: `head.bias_init()` uses default `crop_size=256`. The old gotcha about default 640 was stale — the actual default is 256 which matches PanNuke. If using a different imgsz, `set_model_attributes` calls `bias_init(crop_size=imgsz)` to correct it.

### Ultralytics Integration

14. **InfiniteDataLoader must be wrapped**: `DataLoader` objects are infinite when `shuffle=True`. Never iterate a DataLoader directly.

15. **AMP uses float16**: Forward pass is float16, backward pass is float32. `torch.cuda.amp` handles dtype casting automatically.

16. **Validator double-normalizes**: The validator computes metrics on train_set, but `metrics` method normalizes by len(train_set). For inference on val_set, call `validator(model, dataloader=...)` directly.

17. **XY decode consistency**: Training (`loss.py:403`) returns normalized coords (÷[W,H]). Inference (`head.py:257`) returns pixel coords. Both use same sigmoid+offset+anchor+stride formula.

18. **`steps_per_epoch` computed dynamically (resolved)**: Was hardcoded at 133. Now passed from `RayCastTrainer.get_dataloader()` via `set_steps_per_epoch()` on the criterion. Also set on `_RayCastCriterionWrapper` for the resume path.

19. **O2M/O2O decay schedule fixed (resolved)**: Parent `E2ELoss.decay()` uses `self.updates` (epoch counter) as numerator. Previously `hyp.epochs` was set to `max_epochs × steps_per_epoch` (total steps), but since `updates` is incremented once per epoch, the decay barely moved — o2o stayed at ~20% weight. Now `hyp.epochs = max_epochs` so the schedule spans the full training in epoch units.

20. **TAL o2o loss now uses o2o branch (resolved)**: `loss.py:__call__` was calling `self.one2many.loss(one2one_preds, batch)` which used the o2m assigner (topk=15, topk2=15). Fixed to `self.one2one.loss(one2one_preds, batch)` which uses the o2o assigner (topk=7, topk2=3→1). This was the root cause of all o2o experiments showing no improvement — the o2o branch was never trained with its own assignment.

21. **Hard containment filter removed (tal.py)**: `select_candidates_in_gts` was dead code (never called after removal). All 5376 anchors are candidates; Gaussian decay provides soft spatial weighting. The method has been removed.

22. **`current_epoch` computation fixed**: Was `self.updates / steps_per_epoch` (dividing epoch counter by 166 → giving 0.006 at epoch 1). This meant topk2 annealing, sigma annealing, Hungarian blending, and backbone freeze were never triggered. Now `current_epoch = float(self.updates)`.

23. **piou_loss gradient explosion fixed**: `-log(piou + 1e-7)` produced gradients ~10^7 when piou≈0 in early training, causing AMP GradScaler to skip steps. Now uses `-log(piou.clamp(min=1e-4))` to bound gradient magnitude.

24. **Empty-batch collate fixed**: `raycast_dataset.py` hardcoded `4+32` for fallback tensor shape, which crashes with n_rays=64. Now uses `N_RAYS` constant.

25. **Soft targets normalization fixed**: `loss.py:544` — with `soft_targets=True`, `cls_targets.sum()` sums quality values (~14 for 28 fg) instead of count (28), inflating cls loss ~7x. Now uses `fg_mask.sum()` as denominator when soft_targets is enabled.

28. **DINO denoising only affects assignment, not predictions**: `_inject_denoising_targets()` adds corrupted GT copies to `gt_labels`/`gt_bboxes` before the TAL assigner. The assigner produces more fg assignments (more anchors get cls gradient), but denoising targets are NOT passed to regression — only the real GTs' regression targets are used. The corrupted copies exist solely to provide additional cls training signal.

29. **Standard FL + soft targets is wrong (train25/26)**: Use QFL (`_quality_focal_loss`) when `soft_targets=True`. Standard FL `(1-p_t)^γ` inflates loss for samples near the correct target (when y=0.5, σ=0.5: FL gives MAXIMUM loss, QFL gives zero). Config: `soft_targets_o2o: true` + `focal_gamma_o2o: 2.0` (reused as QFL beta).

30. **max_det reduced to 100**: `RayCastDetect.max_det = 100` (was 300). PanNuke has ~28 cells/image, rare images 50+. 300 was excessive and could include low-confidence false positives.

31. **DINO denoising is dead end for FCN (train28/29)**: In DETR, denoising works via decoder cross-attention that routes queries to specific spatial locations. In FCN, every anchor sees the same feature map — corrupted copies just add confused cls gradient, not contrastive learning. Both train28 (hard targets) and train29 (QFL soft targets) failed: mAP50=0.370 and 0.372 vs train23's 0.517. Denoising config (`dn_num`, etc.) still exists in code but should stay at 0.

32. **Quality head for IoU-aware inference**: 1-channel conv predicting piou per anchor (o2o branch only). At inference: `final_conf = cls × sigmoid(quality)`. Suppresses poorly-localized predictions without any post-processing. Trained with L1 against actual fg_piou. Config: `quality_head_weight` (0=disabled, 1.0=recommended). Head is `one2one_quality_head = copy.deepcopy(quality_head)` — same pattern as separate o2o cls/reg heads.

33. **Quality head is a wash (train30)**: mAP50=0.504 vs train23's 0.517. The head predicts piou for fg anchors but doesn't help cls head suppress false positives — most false positives are confident AND reasonably localized (just duplicates). Quality head is a dead end.

34. **AnchorSelfAttention — linear self-attention on o2o cls features (DEAD END)**: `AnchorSelfAttention(channels=c3, num_heads=4, head_dim=32)` applies multi-head linear attention (Katharopoulos et al., 2020, elu+1 feature map) on c3-dim intermediate cls features before the final nc-projection. Two modes:
    - `self_attention`: Per-scale — each anchor sees all others on the same scale. Applied independently to P2/P3/P4.
    - `cross_scale_attention`: Cross-scale — concatenate all scales (5376 tokens), attend, split back. Each anchor sees all anchors across P2/P3/P4.
    - Both can be combined (per-scale first, then cross-scale). Params: +65K each.
    - Implementation: `forward_head()` splits cv3 Sequential → `[0:-1]` extracts c3 features → attention → `[-1]` projects to nc. Deepcopied for o2o branch. O2M branch has no attention.
    - **Both variants HURT performance** (train31: mAP50=0.462, train32: mAP50=0.465 vs train23's 0.517). Root cause: linear attention with 98.7% bg anchors produces a soft weighted mean that washes out fg cls signal. fg/bg gap dropped from 2.5x to 1.7-1.8x. Code exists but should stay disabled (`self_attention: false`, `cross_scale_attention: false`).

35. **Larger o2o cls head is a wash (train33)**: `cls_channel_scale=2.0` doubles c3 from 128→256. mAP50=0.512 vs train23's 0.517. fg/bg gap narrowed in the wrong direction (2.0x vs 2.5x) — fg confidence dropped more than bg. More capacity without more gradient signal = more uncertainty, not more discrimination. Config: `cls_channel_scale` (default 1.0), `cls_channel_min` (default 0).

36. **AIFI-Lite on P4 is dead end for FCN (train34)**: Single TransformerEncoderLayer on P4 backbone features (16x16=256 tokens, ~790K params). mAP50=0.459 vs train23's 0.517. Same pattern as train31/32 self-attention — feature-level enrichment broadcasts globally-smoothed features to ALL 5376 anchors (98.7% bg), washing out local cls discrimination (fg/bg gap 2.5x→1.6x). In RT-DETR, decoder cross-attention selectively queries enriched features; FCN lacks this mechanism. Implementation was `raycasted/model/blocks/aifi.py` (AIFIBlock) + `raycasted/cfg/yolo26s-aifi-p234.yaml`, both removed in cleanup; model reverted to `yolo26s-run28-p234.yaml`.

37. **Gaussian spatial soft targets collapse fg/bg gap (train35)**: `gaussian_soft_targets=true` with `gaussian_sigma=0.1` gives fg anchors targets exp(-d²/2σ²) instead of 1.0. Same fundamental problem as train25/26/27 piou-based soft targets — spreading fg confidence across 0.3-0.8 instead of sharp 1.0 peak collapses fg/bg gap (2.5x→1.5x). Config: `gaussian_soft_targets` (default false), `gaussian_sigma` (default 0.5). Code exists but should stay disabled.

38. **Higher o2o topk2 severely regresses (train38)**: `o2o_topk2_start=7` (was 3) gives mAP50=0.25 vs train23's 0.517. topk2=7 turns o2o into a weak o2m — diluting 1:1 exclusivity. More fg assignments = less discriminative cls head. topk2=3→1 is near optimal; FCN o2o assignment tuning is exhausted.

39. **PSS head bias must be 2.0, not 0 (train35)**: PSS (Positional Suppression Structure) head predicts per-pixel suppression weight. At inference: `final_conf = cls × sigmoid(pss)`. With bias=0, sigmoid(0)=0.5 halves ALL confidences from step 1, killing recall before the head learns. With bias=2.0, sigmoid(2)≈0.88 gives near-identity at start. Config: `pss_head_weight` (default 0.0, >0 enables). Implementation: `nn.Conv2d(c, 1, 1)` per scale, zero-init weight, bias=2.0, deepcopy for o2o. Loss: BCE on fg anchors (target=1.0 for best anchor per GT, 0.0 for others). **DEAD END**: train36 with stop-gradient showed PSS loss stuck at ~0.627 (barely below random). Stop-gradient prevents cls degradation but PSS head can't learn from frozen features — chicken-and-egg problem.

40. **Root cause across ALL attention/feature-enrichment attempts**: FCN lacks selective access to enriched features. AIFI (train34), self-attention (train31/32), and all feature-level approaches broadcast globally-smoothed features to all 5376 anchors where 98.7% are bg. This washes out local cls discrimination. Only prediction-level interaction (after scoring, on top-K=300 mostly-fg predictions) can create useful competition.

41. **PredictionRefinementAttention — REMOVED** (never trained successfully, code deleted in cleanup). Was self-attention on top-K scored predictions (o2o branch only). After FCN scores all 5376 anchors, top-K=100 by confidence were selected. These K predictions (mostly fg) underwent 1-layer TransformerEncoder self-attention with sincos PE, producing per-prediction suppression weights. At inference: `final_conf = cls × sigmoid(refine)`. ~133K params. Trained with BCE: target=1 for best anchor per GT, 0 for duplicates. Config keys `prediction_refinement_weight` and `prediction_refinement_topk` removed; loss tensor restructured to 7 slots (xy/cls/l1/piou/smooth/distill/aux_xy).

### Eval Script

26. **`raycasted/scripts/eval_pannuke.py` uses streaming metrics**: Rasterizes one image at a time, computes all metrics, frees masks. Peak memory ~2.5GB for 2722 images. Prediction parsing: `pred_confs = det[:, raycast_dim]`, `pred_cls = det[:, raycast_dim + 1]`.

27. **Eval must call `configure_rays(n_rays)` before rasterization**: `_polygons_to_masks_fast` uses `_const.RAY_COS`/`_const.RAY_SIN` (late-binding module attributes).

## Experiment Log

### Baseline Results (no separate head)
Eval (no NMS, conf=0.20): AJI=0.2742, AP@0.5=0.0281, bPQ=0.3041, F1=0.1380, Prec=0.075, Recall=0.865, 759,642 preds vs 65,848 GT (11.5x)

### Separate Head + decay=1.0 + focal_alpha=0.75 (600 epochs)
Val: mAP50=0.411, mAP50-95=0.304, prec=0.417, recall=0.454
Eval (no NMS, conf=0.20): AJI=0.3544, AP@0.5=0.0489, bPQ=0.3748, mPQ=0.0786, F1=0.1521, Prec=0.083, Recall=0.938, 746,450 preds (11.3x)

### Separate Head + decay=0.5 + focal_alpha=0.75 (600 epochs)
Val: mAP50=0.401, mAP50-95=0.296, prec=0.404, recall=0.447
Eval (no NMS, conf=0.20): AJI=0.3442, AP@0.5=0.0440, bPQ=0.3602, mPQ=0.0786, F1=0.1506, Prec=0.082, Recall=0.937, 752,964 preds (11.4x)

**Conclusion: decay=1.0 > decay=0.5.** Higher o2m decay means o2o branch dominates sooner, better for separate o2o cls head training.

### Train21 Results (current_epoch fix + Hungarian blending, 400 epochs)
Val: mAP50=0.448, mAP50-95=0.322, prec=0.46, recall=0.467
Eval (no NMS, conf=0.50): AJI=0.4068, AP@0.5=0.0565, bPQ=0.4762, mPQ=0.1065, F1=0.2163, Prec=0.123, Recall=0.919, 493,583 preds vs 65,848 GT (7.5x)
DIAG: o2o cls fg=0.071, bg=0.031 (barely differentiated). Hungarian weight=0.896 at epoch 400.
**Diagnosis: mAP improved (0.448 vs train13's 0.411) but overprediction persists. Root cause: o2o cls head gets only 28 fg anchors/image (topk2=1) and 825 bg (bg_fg_ratio=3) — 98.7% of anchors get zero cls gradient. Fix must be in training signal, not post-processing.**

### Train22 Results (o2o cls signal boost, best epoch 129, early stopped at 229)
Val: mAP50=0.339, mAP50-95=0.262, prec=0.341, recall=0.439
Eval (no NMS, conf=0.50): AJI=0.422, AP@0.5=0.123, bPQ=0.338, mPQ=0.291, F1=0.444, Prec=0.397, Recall=0.504, **83,674 preds vs 65,848 GT (1.27x)**
Config: `o2o_topk2_anneal_epoch=250`, `bg_fg_ratio_o2o=0`, `soft_targets_o2o=true`, `hungarian_phase2_start=9999`
**Diagnosis: Overprediction solved (7.5x→1.27x!) but recall halved (0.92→0.50) and bPQ dropped (0.476→0.338). All three fixes combined too aggressively — cls gradient overwhelmed regression. F1 doubled, AP@0.5 doubled, mPQ nearly tripled. Need to find the right balance.**

### Train23 Results (bg_fg_ratio_o2o=0 only, 400 epochs)
Val: mAP50=0.517, mAP50-95=0.378, prec=0.550, recall=0.508
Eval (no NMS, conf=0.50): AJI=0.514, AP@0.5=0.347, bPQ=0.491, mPQ=0.403, F1=0.640, Prec=0.660, Recall=0.621, **62,048 preds vs 65,848 GT (0.94x)**
DIAG: o2o cls fg=0.275, bg=0.112 (2.5x gap). Hungarian weight=0.896 at epoch 400.
**Diagnosis: bg_fg_ratio_o2o=0 alone solved overprediction (7.5x→0.94x) while improving every metric except recall. AP@0.5 sextupled, F1 tripled, mPQ quadrupled. Best run so far.**

### Train24 Results (train23 + lambda_suppress=2.0, early stopped ~252)
Val: mAP50=0.449 (regressed from train23's 0.517)
**Diagnosis: Suppress loss spatial repulsion too aggressive for dense PanNuke nuclei — incorrectly suppresses valid adjacent predictions. Killed.**

### Train25 Results (train23 + soft_targets_o2o=true, early stopped at 327, best epoch 227)
Val: mAP50=0.423, mAP50-95=0.321, prec=0.433, recall=0.484
Eval (conf=0.49): AJI=0.478, AP@0.5=0.205, bPQ=0.415, mPQ=0.357, F1=0.544, Prec=0.536, Recall=0.551, 67,646 preds (1.03x)
**Diagnosis: Soft targets with standard focal loss are counterproductive.** They shift the confidence distribution down without improving discriminability. At conf=0.35 barely matches train23's conf=0.49. Root cause: standard FL formula `(1-p_t)^γ` is WRONG for continuous targets — correct is QFL `|y-σ|^β` (GFL, NeurIPS 2020). Killed — moving to QFL.**

### Train26 Results (soft_targets + fixed normalization + Hungarian quality fix, 400 epochs)
Val: mAP50=0.509, mAP50-95=0.378, prec=0.550, recall=0.469
Eval conf=0.49: AJI=0.360, AP@0.5=0.266, bPQ=0.178, F1=0.549, Prec=0.904, Recall=0.394, 28,695 preds (0.44x)
Eval conf=0.35: AJI=0.515, AP@0.5=0.328, bPQ=0.488, F1=0.619, Prec=0.630, Recall=0.608, 63,639 preds (0.97x)
**Diagnosis: Standard FL + soft targets still counterproductive even with fixed normalization. Conf distribution shifts down; at conf=0.35 barely matches train23 at conf=0.49. Confirms QFL is needed.**

### Train27 Results (QFL + max_det=100, soft_targets_o2o=true, 400 epochs)
Val: mAP50=0.514, mAP50-95=0.390, prec=0.543, recall=0.493
Eval conf=0.25: AJI=0.537, AP@0.5=0.317, bPQ=0.548, F1=0.604, Prec=0.551, Recall=0.668, 79,852 preds (1.21x)
Eval conf=0.49: AJI=0.240, bPQ=0.062, F1=0.389, Recall=0.243, 16,543 preds (0.25x)
DIAG o2o: fg/bg gap 3.9x (train23 was 2.5x) — better discrimination in training
**Diagnosis: QFL shifts conf distribution down less than FL, but same fundamental problem — soft targets spread fg scores across 0.3-1.0 instead of clustering at ~1.0. No clean decision boundary.**

### Train28 Results (DINO denoising + hard targets + max_det=100, early stopped at 296)
Val: mAP50=0.370, mAP50-95=0.287, prec=0.435, recall=0.414. Best epoch 225.
DIAG o2o: fg jumped from ~290 to ~1950 per batch. cls loss nearly doubled.
**Diagnosis: Denoising floods o2o with fg anchors, but hard 0/1 targets give no quality gradient.**

### Train29 Results (DINO denoising + QFL + max_det=100, early stopped at 296)
Val: mAP50=0.372, mAP50-95=0.288, prec=0.448, recall=0.401. Best epoch 196.
**Diagnosis: Denoising + QFL together still failed. mAP barely above train28 (0.370). The extra fg from denoising doesn't create useful contrastive signal even with QFL. Denoising is a dead end for FCN.**

### Train30 Results (quality_head_weight=1.0, 400 epochs)
Val: mAP50=0.504, mAP50-95=0.376, prec=0.532, recall=0.509
O2O DIAG: fg=0.252, bg=0.105 (2.4x gap — same as train23's 2.5x)
**Diagnosis: Quality head is a wash.** mAP slightly worse (0.504 vs 0.517). fg/bg discrimination unchanged. The head predicts piou but doesn't help cls suppress false positives — most FPs are confident AND well-localized (just duplicates).

### Train31 Results (per-scale self-attention, 600 epochs)
Val: mAP50=0.462, mAP50-95=0.347, prec=0.470, recall=0.487
O2O DIAG: fg=0.158, bg=0.095 (1.7x gap — worse than train23's 2.5x)
**Diagnosis: Per-scale self-attention HURTS.** Linear attention averages sparse fg into bg noise. fg/bg gap collapsed from 2.5x to 1.7x. Dead end.

### Train32 Results (cross-scale self-attention, 600 epochs)
Val: mAP50=0.465, mAP50-95=0.349, prec=0.489, recall=0.477
O2O DIAG: fg=0.156, bg=0.088 (1.8x gap — same problem as train31)
**Diagnosis: Cross-scale same problem.** More tokens (5376) makes averaging slightly worse. Dead end.

### Train33 Results (cls_channel_scale=2.0, 400 epochs)
Val: mAP50=0.512, mAP50-95=0.377, prec=0.553, recall=0.509
O2O DIAG: fg=0.198, bg=0.099 (2.0x gap — worse than train23's 2.5x)
**Diagnosis: Larger o2o cls head is a wash.** fg/bg gap narrowed in wrong direction (2.0x vs 2.5x) — fg confidence dropped more than bg. More capacity without more gradient signal = more uncertainty. Dead end.

### Train34 Results (AIFI-Lite on P4, 600 epochs)
Val: mAP50=0.459, mAP50-95=0.344, prec=0.477, recall=0.489
O2O DIAG: fg=0.120, bg=0.076 (1.6x gap — worse than train23's 2.5x)
**Diagnosis: AIFI-Lite is a dead end for FCN.** Same pattern as train31/32 — feature-level enrichment broadcasts globally-smoothed features to all 5376 anchors, washing out local cls discrimination (fg/bg gap 2.5x→1.6x). In RT-DETR, decoder cross-attention selectively queries enriched features; FCN lacks this mechanism.

### Train35 Results (PSS head + Gaussian soft targets, early stopped epoch 195)
Val: mAP50=0.424, mAP50-95=0.346, prec=0.801, recall=0.053
**CATASTROPHIC**: Recall collapsed from 0.62 (train23) to 0.053. Gaussian soft targets spread fg confidence across 0.3-0.8 (collapsing fg/bg gap 2.5x→1.5x). PSS head with bias=0 (sigmoid(0)=0.5) halved all confidences from step 1. PSS head bias fixed to 2.0 for future runs; Gaussian soft targets are a dead end.

### Train36 Results (PSS head + stop-gradient, still running)
PSS loss stuck at ~0.627 (barely below random 0.693), mAP50 ~0.370. Stop-gradient prevents cls degradation but PSS head can't learn from frozen features — chicken-and-egg problem. PSS head is a dead end (both with and without stop-gradient).

### Train38 Results (o2o_topk2_start=7, early stopped epoch 216)
Val: mAP50=0.25
**Diagnosis: Higher o2o topk2 severely regresses.** topk2=7 turns o2o into a weak o2m — diluting 1:1 exclusivity that makes o2o effective. More fg assignments = less discriminative cls head. topk2=3→1 is near optimal; FCN o2o assignment tuning is exhausted.

### Pending Experiments
- RT-DETR + RayCast — IMPLEMENTED, smoke tested, NOT trained

## Testing

**Test entry points:**
- `tests/phase_6/` — 27 tests via pytest
- Single test file: `uv run python -m pytest tests/phase_6/test_tal.py -x -v`

**Pre-existing failures:** 7 tests fail (Hungarian assigner tests, loss constructor lambda_l1 14 vs 25, smoothness annealing). 19 pass. These are pre-existing and unrelated to recent changes. Total: 28 tests (including the decay schedule test).

**Test file naming:**
- Phase 1: TAL assignment tests (tal.py)
- Phase 2: Head tests (head.py)
- Phase 3: Loss tests (loss.py)
- Phase 4: Trainer tests (train.py)
- Phase 5: Dataset tests (raycast_dataset.py)
- Phase 6: E2E integration tests

## Run Tracking

**All training runs must be documented in `docs/runs/`** — one markdown file per run (e.g., `docs/runs/train13.md`, `docs/runs/train18.md`). Each run file should include:

- Run number and date
- Config changes from previous run (diff of `pannuke.yaml` or explicit list)
- Key hyperparameters (especially anything non-default)
- Training results: val mAP50, mAP50-95, precision, recall at best epoch
- Eval results: AJI, bPQ, mPQ, AP@0.5, F1, prediction count vs GT
- Diagnosis: what worked, what didn't, and why
- Next steps / follow-up experiments

This ensures every run is reproducible and we can trace the evolution of config decisions.

## Style & Conventions

- Follow ruff formatting and linting (`uv run ruff check . && uv run ruff format .`)
- Use `uv run python` for all scripts
- Prefer executable source of truth over prose (configs, scripts > docs)
- When docs conflict with config/scripts, trust the executable source
