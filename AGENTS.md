# AGENTS.md

This file helps OpenCode agents work efficiently in the RayCastED repository.

## Essential Commands

```bash
# Install dependencies
uv sync

# Lint and format (must run together)
uv run ruff check .
uv run ruff format .

# Run tests (no test runner — direct Python execution)
uv run python tests/phase_0_5/test_round_trip.py
uv run python tests/phase_6/test_loss.py

# Full pipeline (ETL + training)
uv run python -m raycasted.pipeline --config main/dataset.yaml --output output/ --epochs 100

# Individual pipeline stages
uv run python -m raycasted.pipeline --config main/dataset.yaml --output output/ --stage ingest --dataset PUMA
uv run python -m raycasted.pipeline --config main/dataset.yaml --output output/ --stage transform
uv run python -m raycasted.pipeline --config main/dataset.yaml --output output/ --stage train --epochs 50
```

## Training Workflow

**The user trains on a separate GPU device.** Do NOT launch training commands (e.g. `--stage train`) from this agent session. Instead:

1. Update `main/pannuke.yaml` (or the relevant config) with the new model YAML path.
2. Inform the user the config is ready — they will run training themselves, typically:
   ```bash
   uv run python -m raycasted.pipeline --config main/pannuke.yaml --output output/<RunName> --epochs 500 --device 0 --stage train
   ```
3. The pipeline uses ultralytics' built-in resume from `last.pt` — if training is interrupted, re-running the same command resumes from the last checkpoint.

## Code Style

**Ruff config** (`pyproject.toml`): line length 120, single quotes, Google-style docstrings, isort with `raycasted` as first-party.

Always run `uv run ruff check . && uv run ruff format .` before committing.

## Architecture Overview

**Goal:** Convert YOLOv8 (bounding-box detector) into a raycast polygon detector for cell detection in histopathology images.

**Data flow:**
```
YAML config (main/dataset.yaml)
    ↓  ETLConfig + IngestionOrchestrator
.npz files [image + raycast annotations, pixel space]
    ↓  TransformOrchestrator (SpatialChunker + NormalizerAndPadder)
.npz tiles [content_h, content_w preserved]
    ↓  RayCastTileDataset
[B, 3, H, W] + [M, 36] labels (normalised)
    ↓  RayCastTrainer
Trained weights
    ↓  RayCastPredictor
[N, 32, 2] polygon vertices (pixel space)
```

## Critical Implementation Details

### Annotation Format
All stages use a single array format — no conversion between ETL and model:
```
[class_id, cx, cy, d_1, ..., d_32]   shape: (N, 35), float32, pixel space
```
Collated batch format adds leading `batch_idx`: shape `(sum_M, 36)`.

Normalisation (divide by `crop_size`, default 640) happens **only** in `RayCastTileDataset._normalise()`. Denormalisation at inference must use `crop_size` from `model.training_args['crop_size']` — never hardcoded.

### Module Boundaries

**`raycasted/data/etl/ops/`** — Single source of truth for ALL geometry logic. Every caller imports from here; nothing is reimplemented elsewhere. PyTorch variants use lazy imports (`import torch` inside function body) for ETL safety.

**`raycasted/data/etl/utils/constants.py`** — Angular convention, permutation indices, format indices. Import from here; never recompute inline. Key constants: `RAY_ANGLES`, `CLASS_IDX=0`, `CX_IDX=1`, `CY_IDX=2`, `RAY_START_IDX=3`, `RAY_END_IDX=35`.

**`raycasted/model/`** — Prediction head and training package. Subclasses `ultralytics.nn.modules.head.Detect` rather than modifying ultralytics in-place. Uses `InfiniteDataLoader` (not plain `DataLoader`) for trainer compatibility.

### Structural Note

The implementation nests modules under `etl/`:
- `raycasted/data/etl/ops/` (geometry)
- `raycasted/data/etl/utils/` (config, constants)
- `raycasted/data/etl/loader/` (dataset)

All import paths must use these actual locations.

## Known Gotchas

### Parallel Processing
- **DEADLOCK**: `ProcessPoolExecutor` with default `fork` start method deadlocks with OpenMP-backed libraries (`cv2`, `polars`). Fix: use `multiprocessing.get_context('spawn')` for all process pools.
- **PanNuke parallelism**: Only 3 registry rows (folds), so orchestrator-level parallelism uses ≤3 cores. `ParquetIngestor` handles internal ROI-level parallelism.

### GPU Training
- **AMP dtype mismatches**: `torch.cdist` and IoU computations fail with mixed float16/float32. Cast to `.float()` before these operations.
- **Ignore class filtering**: `RayCastTileDataset` must filter `class_id=255` (Ignore) annotations to prevent index-out-of-bounds in assigner.
- **Validator double normalization**: `RayCastTileDataset` returns float32 [0,1] images, but `DetectionValidator.preprocess()` divides by 255. Override to skip /255 to prevent mAP collapse.

### Ultralytics Integration
- **DataLoader**: Must use `ultralytics.data.build.InfiniteDataLoader` — Ultralytics calls `train_loader.reset()` after training.
- **Plotting crashes**: Ultralytics' `plot_images()` and `plot_predictions()` expect 4-dim bboxes but get 34-dim polygon data. Override as no-ops in trainer/validator.
- **Bias initialization**: `RayCastDetect.bias_init()` must use `crop_size` not `stride` for normalised-space predictions.

### E2E Dual-Assignment Architecture (NMS-Free)
- **CRITICAL**: RayCastED uses `E2ELoss` for NMS-free detection.
- **Architecture**: `one2many` branch (dense supervision) + `one2one` branch (NMS-free enforcement).
- **Assignment**: Both branches use `RayCastAssigner` with Polar-IoU for matching.
- **NMS-free requirement**: `one2one.assigner.topk=1` (exactly 1 positive anchor per GT for NMS-free inference).
- **Config parameter**: Use `tal_topk` (NOT `assigner_topk`) - this correctly propagates to both branches.
- **Default values**: `tal_topk=13`, `assigner_radius_scale=1.5`, `focal_loss=false` (proven baseline).
- **Validation**: Check console output for `✓ E2E NMS-free: o2m.topk=13, o2o.topk=1` during training.
- **Weight decay**: Parent `E2ELoss` decays `o2m` weight from 0.8→0.1 over training. `RayCastE2ELoss` must set `hyp.epochs` on both branches to match actual `max_epochs`, otherwise the schedule collapses (e.g. `hyp.epochs=100` with `max_epochs=200` starves one2many for the entire second half of training).
- **topk2 (secondary filtering)**: Ultralytics' NMS-free mechanism uses a two-stage assignment: `select_topk_candidates` picks `topk` anchors, then `select_highest_overlaps` checks `topk2 != topk` and keeps only `topk2` best. RayCastED must set `one2one.topk2=1` (NOT `one2one.topk=1`). Using `topk=1` directly skips the candidate pool, giving the assigner no choice. Use `topk=max(tal_topk//2, 7)` to provide a candidate pool. For `one2many`, set `topk2=topk` to disable secondary filtering.
- **IoU-aware scoring**: The centerness branch is trained to predict **Polar-IoU** (not centered-ness). At inference, `score = cls * sigmoid(centerness)` becomes quality-aware confidence, matching LSP-DETR's `score = cls * IoU` strategy. This is critical for mAP@0.5 — without it, poorly-shaped polygons get high confidence and corrupt the PR curve ordering.
- **Confidence threshold**: With IoU-aware scoring, scores are naturally lower (multiplied by IoU ≈ 0.3-0.7 vs centerness ≈ 0.8-0.95). Use `conf=0.20` (not the default 0.25) for optimal mAP@0.5. This was validated on PanNuke Fold3: conf=0.20 gives mAP@0.5=0.391, F1=0.641 vs conf=0.25 gives mAP@0.5=0.387, F1=0.636.

### Geometry Operations
- **XY decode mismatch**: Training and inference must use same decode formula: `(sigmoid * 2.0 - 0.5 + anchor) * stride`.
- **Val losses NaN**: Force `overlaps` tensor to float32 in assigner — `eps=1e-9` underflows to 0 in float16.
- **Centroid collapse**: `Huber(delta=0.01)` on normalised coordinates produces tiny gradients. Use `lambda_xy=50.0, delta=1.0` for stronger centroid loss.
- **NMS-free loss weights**: For `one2one.topk=1`, use LSP-DETR-inspired weighting:
  - `lambda_l1=5.0` (very strong direct ray supervision)
  - `lambda_piou=0.5` (minimal IoU, mainly for ranking)
  - This compensates for sparse assignment while maintaining NMS-free property.

### LSP-DETR Baseline (PanNuke Fold3)
Reference values from LSP-DETR author evaluation (source: `docs/ablation.md`):
| Metric | LSP-DETR |
|--------|----------|
| AJI | 0.677 |
| AP@0.5 | 0.691 |
| AP@0.5:0.95 | 0.441 |
| AP@0.7 | 0.563 |
| AP@0.9 | 0.100 |
| Precision | 0.862 |
| Recall | 0.791 |
| F1 (centroid) | 0.825 |
| bDQ | 0.803 |
| bSQ | 0.807 |
| bPQ | 0.657 |
| bMDQ | 0.811 |
| bMSQ | 0.811 |
| bMPQ | 0.666 |
| mDQ | 0.581 |
| mSQ | 0.672 |
| mPQ | 0.476 |
| mMDQ | 0.481 |
| mMSQ | 0.676 |
| mMPQ | 0.481 |
⚠️ LSP-DETR F1 is centroid-based (Euclidean distance threshold). RayCastED ablation F1 is mask-based (DQ from PQ). Not directly comparable.

## Testing Strategy

Tests are plain `assert`-based scripts organized by phase:

- **Phase 0.5**: `tests/phase_0_5/test_round_trip.py` — geometry round-trip tests (CPU-only)
- **Phase 5-6**: `tests/phase_6/test_loss_gpu.py` — GPU loss/assigner tests
- **Phase 10**: `tests/phase_10/test_train_gpu.py` — GPU training integration
- **Phase 11**: `tests/phase_11/test_pipeline_gpu.py` — end-to-end pipeline

GPU-bounded tests require CUDA — all others are CPU-only or require only real data access.

## Configuration

**Single source of truth**: `docs/project.md` — read it before modifying any module.

**ETL config**: `main/dataset.yaml` contains all dataset-specific settings. `annotation_type` is a global-only setting — one pipeline run uses one annotation type for all datasets.

**Size unification**: Two validated Pydantic fields:
- `max_size`: ETL tile size (e.g., 1024)
- `crop_size`: Model input size (e.g., 640)

Pipeline defaults `imgsz` from config's `crop_size`.

## Deferred Features

- `MatInstIngestor._extract_raycast_annotations()` is a stub (`NotImplementedError`)
- H&E overlay validation
- Zero-ray fraction < 1% validation
- Ingestor diagnostic counters not yet wired
- MLflow logging of annealing values requires custom callback

### XY Plateau (Scratch Training)
- **Root cause**: Backbone feature poverty — the head cannot learn centroid regression from random backbone features (proven by simulation in `docs/xy_plateau_analysis.md`). Chicken-and-egg: head needs spatial features, backbone needs xy gradient to learn them.
- **No pretrained weights**: This is a new architecture with a custom RayCast head. YOLO COCO pretrained weights have negligible impact because the domain shift from natural images to histopathology negates most benefits. `pretrained_backbone: null` is the default.
- **What works**: Classification converges first (easier task, only needs object presence), then backbone features become spatially informative, then xy regression breaks through the plateau.
- **Assigner warmup** (`assigner_warmup_epochs=50`): First N epochs use Gaussian centroid-distance instead of Polar-IoU for assignment. Prevents garbage matches from meaningless early ray predictions.
- **Lambda_l1 warmup suppression**: During warmup `lambda_l1=0.1` (floor), jumps to `25.0` after warmup.
- **Rejected hypotheses**: Muon orthogonalization does NOT suppress xy (lambda_xy=500 makes xy gradient dominate 99.9% of head weight matrix). Sigmoid saturation and decode range are not the bottleneck.
- **Current config**: `lambda_xy=500.0`, `optimizer=MuSGD`, `lr0=0.01`, `cos_lr=true`, `tal_topk=20`, `assigner_warmup_epochs=50`.

## Deployment

Target: NVIDIA Jetson (Orin/Xavier) via ONNX → TensorRT FP16 inference.
