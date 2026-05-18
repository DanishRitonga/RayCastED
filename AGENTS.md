# AGENTS.md - RayCastED Instructions

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
- `tests/phase_6/` — 27 pytest tests

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

1. **bg_cls_decay double-application**: `raycasted/model/train.py:361` applies bg_cls_decay twice — once in the loop, once in the partial function. This over-regularizes background classes, causing them to be treated as easy negatives.

2. **focal_alpha=0.5 causes mode collapse**: When alpha=0.5, background gets 3x more weight than foreground. Combined with the nc=80 issue (75 ghost channels), this overwhelms the 5 real classes.

3. **class_weights failure**: `class_weights` in loss.py is ignored during warmup (epochs 0-50) because the assigner uses centroid-distance similarity, which doesn't use cls scores.

4. **Shared cv3 fix (resolved)**: O2O branch now has separate `one2one_cv3` cls head. `fuse()` sets `cv2=cv3=None`, keeping only o2o heads. Shared head caused 11.5x overprediction.

5. **Separate head alone doesn't fix overprediction**: Even with separate o2o cls head, model still produces 11.3x overprediction (746k preds vs 66k GT). The o2o head needs more positive signal during training — topk2 annealing (3→1) addresses this.

6. **Hard binarization destroys quality signal**: `loss.py:475` sets `cls_targets[cls_targets > 0] = 1.0`. All positives get same target regardless of match quality. Soft targets for o2o may help.

### Assignment & Warmup

7. **warmup sigma must be configurable**: Sigma is hardcoded at 0.15 in `tal.py:get_box_metrics`. Always thread sigma through loss.py → tal.py and update pannuke.yaml.

8. **assigner_radius_scale must match model geometry**: Default fallback [8,16,32] in loss.py is wrong for P2-P3-P4 architecture. Use model's stride values instead.

9. **dynamic topk masks garbage**: `select_topk_candidates` in tal.py masks out zero-metric entries. You can increase tal_topk to get more candidates through the containment filter.

10. **3-phase Hungarian blending**: Hungarian assigner for o2o blended via smooth schedule in `RayCastE2ELoss`. Phase 1 (0→100): pure TAL. Phase 2 (100→250): Hungarian ramps 0→0.9, TAL fades. Phase 3 (250+): Hungarian at 0.9, TAL at 0.1. Config: `hungarian_phase2_start`, `hungarian_phase3_start`, `hungarian_max_weight`, `hungarian_cost_{class,centroid,ray}`. Implementation: `_compute_hungarian_o2o_loss()` temporarily swaps the o2o assigner.

11. **Topk2 annealing**: `o2o_topk2_start` (default 3) and `o2o_topk2_anneal_epoch` (default 120) linearly anneal topk2 from start→1 over [anneal_epoch, max_epochs]. Inspired by One-to-Few (CVPR 2023). Config in pannuke.yaml.

### Model Construction

12. **nc must be set at construction time**: Pass `nc=self.data['nc']` in `get_model()` (train.py). If nc is set after construction, cv3 conv layers stay at 80 channels.

13. **bias_init default (640) wrong**: `head.bias_init()` uses a hardcoded default. After the first `set_model_attributes` call, it's harmless but wrong during initial construction.

### Ultralytics Integration

14. **InfiniteDataLoader must be wrapped**: `DataLoader` objects are infinite when `shuffle=True`. Never iterate a DataLoader directly.

15. **AMP uses float16**: Forward pass is float16, backward pass is float32. `torch.cuda.amp` handles dtype casting automatically.

16. **Validator double-normalizes**: The validator computes metrics on train_set, but `metrics` method normalizes by len(train_set). For inference on val_set, call `validator(model, dataloader=...)` directly.

17. **XY decode consistency**: Training (`loss.py:403`) returns normalized coords (÷[W,H]). Inference (`head.py:257`) returns pixel coords. Both use same sigmoid+offset+anchor+stride formula.

18. **Hardcoded `steps_per_epoch=133`** (`loss.py:744`): Used for smooth loss annealing and aux_xy decay schedules. If actual steps/epoch differs, schedules activate at wrong times. TODO: compute dynamically.

### Eval Script

19. **`main/eval_pannuke.py` uses streaming metrics**: Rasterizes one image at a time, computes all metrics, frees masks. Peak memory ~2.5GB for 2722 images. Prediction parsing: `pred_confs = det[:, raycast_dim]`, `pred_cls = det[:, raycast_dim + 1]`.

20. **Eval must call `configure_rays(n_rays)` before rasterization**: `_polygons_to_masks_fast` uses `_const.RAY_COS`/`_const.RAY_SIN` (late-binding module attributes).

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

### Pending Experiments
- Topk2 annealing (3→1 at epoch 120) — training in progress
- 3-phase Hungarian blending (code implemented, needs training run)
- Soft cls targets for o2o only
- Higher inference conf (0.5) — tested, still 609k preds at conf=0.5. Training fix needed.
- Reduce max_det from 300 to 50-100

## Testing

**Test entry points:**
- `tests/phase_6/` — 27 tests via pytest
- Single test file: `uv run python -m pytest tests/phase_6/test_tal.py -x -v`

**Pre-existing failures:** 7 tests fail (Hungarian assigner tests, loss constructor lambda_l1 14 vs 25, smoothness annealing). 19 pass. These are pre-existing and unrelated to recent changes.

**Test file naming:**
- Phase 1: TAL assignment tests (tal.py)
- Phase 2: Head tests (head.py)
- Phase 3: Loss tests (loss.py)
- Phase 4: Trainer tests (train.py)
- Phase 5: Dataset tests (raycast_dataset.py)
- Phase 6: E2E integration tests

## Style & Conventions

- Follow ruff formatting and linting (`uv run ruff check . && uv run ruff format .`)
- Use `uv run python` for all scripts
- Prefer executable source of truth over prose (configs, scripts > docs)
- When docs conflict with config/scripts, trust the executable source
