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
- `raycasted/model/head.py` — classification + ray head, bias_init, parse_output
- `raycasted/model/loss.py` — loss computation, assignment integration
- `raycasted/model/tal.py` — assignment logic (Hungarian + greedy one2one)
- `raycasted/data/etl/utils/constants.py` — N_RAYS, ANGLE state (module-level)

**Execution flow:**
```
train.py → get_model() (constructs model with nc=self.data['nc'])
  ↓
RayCastDetectionLoss (loss.py) → RayCastAssigner (tal.py) → Hungarian or one2one assignment
  ↓
Forward pass → parse_output (head.py) → decode for inference
```

## Training Config Threading Pattern

When adding a new config parameter:
1. Add to `_RayCastCriterionWrapper.__call__` config parsing (train.py:~180)
2. Pass to `RayCastE2ELoss.__init__` (loss.py:~575)
3. Pass to `RayCastDetectionLoss.__init__` (loss.py:~280)
4. Pass to assigner constructors in loss.py (~661, ~676)
5. Add to main/pannuke.yaml
6. Run: `ruff check && ruff format && pytest tests/phase_6/ -x -q`

## Critical Gotchas

### Classification Collapse

1. **bg_cls_decay double-application**: `raycasted/model/train.py:361` applies bg_cls_decay twice — once in the loop, once in the partial function. This over-regularizes background classes, causing them to be treated as easy negatives.

2. **focal_alpha=0.5 causes mode collapse**: When alpha=0.5, background gets 3x more weight than foreground. Combined with the nc=80 issue (75 ghost channels), this overwhelms the 5 real classes.

3. **class_weights failure**: `class_weights` in loss.py is ignored during warmup (epochs 0-50) because the assigner uses centroid-distance similarity, which doesn't use cls scores.

### Assignment & Warmup

4. **warmup sigma must be configurable**: Sigma is hardcoded at 0.15 in `tal.py:get_box_metrics`. This permissive Gaussian matching allows anchors 30-40px away from GT centroids to get high alignment scores, creating shallow gradient wells. Always thread sigma through loss.py → tal.py and update pannuke.yaml.

5. **assigner_radius_scale must match model geometry**: Default fallback [8,16,32] in loss.py is wrong for P2-P3-P4 architecture. Use model's stride values instead.

6. **dynamic topk masks garbage**: `select_topk_candidates` in tal.py masks out zero-metric entries (lines 398-401). You can increase tal_topk to get more candidates through the containment filter.

### Model Construction

7. **nc must be set at construction time**: Pass `nc=self.data['nc']` in `get_model()` (train.py:364). The assertion guard (line 584) ensures head.nc == self.model.nc, but if you set nc after construction, cv3 conv layers stay at 80 channels.

8. **bias_init default (640) wrong**: `head.bias_init()` uses a hardcoded default. After the first `set_model_attributes` call, it's harmless but wrong during initial construction.

### Ultralytics Integration

9. **InfiniteDataLoader must be wrapped**: `DataLoader` objects are infinite when `shuffle=True`. Never iterate a DataLoader directly — wrap in `torch.utils.data.InfiniteDataLoader` if you need to access iterables.

10. **AMP uses float16**: Forward pass is float16, backward pass is float32. The model must be fully FP16 compatible. `torch.cuda.amp` handles dtype casting automatically.

11. **Validator double-normalizes**: The validator computes metrics on train_set, but `metrics` method normalizes by len(train_set). For inference on val_set, call `validator(model, dataloader=...)` directly.

12. **XY decode mismatch**: Loss decode uses sigmoid + offset. Inference decode must use the same formula, or predictions will be wrong.

13. **Hardcoded `steps_per_epoch=133`** (`loss.py:744`): Used for smooth loss annealing and aux_xy decay schedules. If actual steps/epoch differs (different dataset/batch size), schedules activate at wrong times. TODO: compute dynamically from dataset size and batch size.

14. **Shared cv3 overprediction fix** (resolved): The o2o branch now has a separate `one2one_cv3` cls head (not shared with o2m). The shared head caused 11.5x overprediction because o2m's dense positives (topk=15) taught the shared cls head to fire high scores for many anchors per GT. `fuse()` now sets `cv2=cv3=None`, keeping only o2o heads for inference.

## Testing

**Test entry points:**
- `tests/phase_6/` — 27 tests via pytest
- Single test file: `uv run python -m pytest tests/phase_6/test_tal.py -x -v`

**Test file naming:**
- Phase 1: TAL assignment tests (tal.py)
- Phase 2: Head tests (head.py)
- Phase 3: Loss tests (loss.py)
- Phase 4: Trainer tests (train.py)
- Phase 5: Dataset tests (raycast_dataset.py)
- Phase 6: E2E integration tests

## Style & Conventions

- Follow ruff formatting and linting (`uv run ruff check . && ruff format .`)
- Use `uv run python` for all scripts
- Prefer executable source of truth over prose (configs, scripts > docs)
- When docs conflict with config/scripts, trust the executable source
