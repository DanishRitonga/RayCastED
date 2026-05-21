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
- `main/eval_pannuke.py` — streaming eval (AJI, bPQ, mPQ, AP, F1)

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

### Eval Script

22. **`main/eval_pannuke.py` uses streaming metrics**: Rasterizes one image at a time, computes all metrics, frees masks. Peak memory ~2.5GB for 2722 images. Prediction parsing: `pred_confs = det[:, raycast_dim]`, `pred_cls = det[:, raycast_dim + 1]`.

23. **Eval must call `configure_rays(n_rays)` before rasterization**: `_polygons_to_masks_fast` uses `_const.RAY_COS`/`_const.RAY_SIN` (late-binding module attributes).

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

### Pending Experiments
- Isolate fixes one at a time: try only bg_fg_ratio_o2o=0 (no soft targets)
- Or try bg_fg_ratio_o2o=10 instead of 0 (partial relaxation)
- Keep all three but increase regression lambdas to balance
- Suppress loss (lambda_suppress) available — test on stable config
- Reduce max_det from 300 to 50-100

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
