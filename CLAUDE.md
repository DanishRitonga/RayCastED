# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` (Python 3.11).

```bash
# Install dependencies
uv sync

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Run a script
uv run python raycasted/data/etl/utils/constants.py
```

No test runner is configured yet. Tests are plain `assert`-based scripts:

```bash
# Phase 0.5 round-trip tests
uv run python tests/phase_0_5/test_round_trip.py
```

**Ruff config** (`pyproject.toml`): line length 120, single quotes, Google-style docstrings, isort with `raycasted` as first-party.

## Architecture

**Goal:** Convert Ultralytics YOLOv26 (bounding-box detector) into a raycast polygon detector for Tumour-Infiltrating Lymphocyte (TIL) detection in histopathological WSIs. The authoritative specification is `docs/project.md` — read it before modifying any module.

### Pipeline

```
YAML config (main/dataset.yaml)
    ↓  ETLConfig (Pydantic) + IngestionOrchestrator (ingestors/ingestion_orchestrator.py)
Ingestors → .npz files  [image + raycast annotations, pixel space]
    ↓  TransformOrchestrator (SpatialChunker + NormalizerAndPadder)
.npz tiles  [content_h, content_w preserved]
    ↓  RayCastTileDataset
[B, 3, H, W] + [M, 36] labels (normalised)
    ↓  RayCastTrainer (Phase 10) wires RayCastDetect + RayCastE2ELoss + RayCastAssigner
Trained weights
    ↓  RayCastPredictor (Phase 7)
[N, 32, 2] polygon vertices  (pixel space)
```

### Key modules

**`raycasted/model/`** — Prediction head and training package (Phases 4–10). Subclasses `ultralytics.nn.modules.head.Detect` rather than modifying ultralytics in-place. Contains `RayCastDetect`, `RayRefinementBlock`, `register_raycast_head()` for namespace injection, `RayCastDetectionLoss`, `RayCastAssigner`, `RayCastE2ELoss`, `RayCastPredictor`, `RayCastValidator`, and `RayCastTrainer`. Uses `InfiniteDataLoader` (not plain `DataLoader`) for trainer compatibility. Files: `head.py`, `register.py`, `loss.py`, `tal.py`, `predict.py`, `val.py`, `train.py`.

**`raycasted/data/etl/ops/`** — Single source of truth for ALL geometry logic. Every caller imports from here; nothing is reimplemented elsewhere. PyTorch variants (`polar_iou_torch`, `angular_smoothness_loss_torch`) use **lazy imports** (`import torch` inside the function body) so this package is safe to import without PyTorch in ETL environments. `decode_pred_xy` is NOT here — it is a method of `RayCastDetectionLoss` only.

**`raycasted/data/etl/utils/constants.py`** — Angular convention, permutation indices, and format indices. Import from here; never recompute inline. Key constants: `RAY_ANGLES`, `FLIP_H_IDX`, `FLIP_V_IDX`, `ROT_INDICES`, `CLASS_IDX=0`, `CX_IDX=1`, `CY_IDX=2`, `RAY_START_IDX=3`, `RAY_END_IDX=35`.

**`raycasted/data/etl/utils/config.py`** — `ETLConfig` wraps a YAML file via Pydantic (`ETLConfigModel`). `annotation_type` is a **global-only** setting — one pipeline run uses one annotation type for all datasets. To ingest with different types, run separate configs.

**`raycasted/data/etl/ingestors/_base.py`** — `BaseDataIngestor` handles file discovery, split assignment, and MPP scaling. Label translation is two-step: raw dataset string → namespace-standard string → global integer (via `namespace_map` + `global_cell_map`). The split column in the Polars registry is always `'split'`.

### Structural note

The plan (`docs/project.md §7`) specifies `raycasted/data/ops/`, `raycasted/data/utils/`, and `raycasted/data/loader/` as top-level siblings of `etl/`. The actual implementation nests them under `etl/`: `raycasted/data/etl/ops/`, `raycasted/data/etl/utils/`, `raycasted/data/etl/loader/`. All import paths must use the actual locations.

### Annotation format

All stages share a single array format — no conversion between ETL and model:

```
[class_id, cx, cy, d_1, ..., d_32]   shape: (N, 35), float32, pixel space
```

Collated batch format adds a leading `batch_idx` column: shape `(sum_M, 36)`.

Normalisation (divide by `crop_size`, default 640, configured in ETL YAML `global_settings.crop_size`) happens **only** in `RayCastTileDataset._normalise()`. Denormalisation at inference must use `crop_size` read from `model.training_args['crop_size']` — never hardcoded.

### .npz schema

Post-`TransformOrchestrator` tiles contain: `image` (uint8, HWC), `annotations` (float32, N×35), `tissue` (int32), `content_h` (int32), `content_w` (int32). Load with backward-compat key: `data.get('annotations', data.get('bboxes'))`.

### Known issues

- `MatInstIngestor._extract_raycast_annotations()` is a stub (`NotImplementedError`) — deferred, requires contour extraction from instance maps.
- Phase 0.5 deferred items (require real datasets): H&E overlay, `d_i` ≤ centroid-to-edge check, zero-ray fraction < 1%.
- Ingestor diagnostic counters not yet wired: `fallback_counter` not passed to `polygon_to_raycast()`, zero-ray logging not implemented. Must be added before first real-data run (see §12.4, NOTE-04).
- MLflow logging of `lambda_smooth` and `o2m_weight` annealing values requires a custom callback (per-term losses are logged automatically via Ultralytics' built-in MLflow integration when MLflow is installed).
- `GeoJSONIngestor` fails on 4 PUMA ROIs with `'LineString' object has no attribute 'x'` — degenerate geometries not handled.

### Implementation order

Follow `docs/project.md §8` strictly — each phase depends only on phases above it:

```
Phase 0   — ops/ + constants  ✓
Phase 0.5 — visual round-trip tests  ✓
Phase 1   — raycast extraction in all 3 ingestors  ✓
Phase 1.5 — IngestionOrchestrator  ✓
Phase 2   — SpatialChunker / NormalizerAndPadder / TransformOrchestrator patches  ✓
Phase 3   — RayCastTileDataset + collate_fn  ✓
Phase 4   — Model head (RayRefinementBlock, RayCastDetect, 34-dim output)  ✓
Phase 5   — RayCastAssigner (masked pairwise IoU)  ✓
Phase 6   — RayCastDetectionLoss (5 terms) + RayCastE2ELoss  ✓
Phase 7   — RayCastPredictor + RayCastAnnotator  ✓
Phase 8   — RayCastValidator  ✓
Phase 9   — ONNX export + TensorRT deployment (NVIDIA Jetson)
Phase 10  — Training integration (RayCastTrainer + YAML config + MLflow)  ✓
Phase 11  — Pipeline orchestrator (RayCastPipeline)  ✓
```

**Phase 4 approach:** Subclasses `ultralytics Detect` in `raycasted/model/head.py` instead of modifying `ultralytics/` in-place. Registration via `register_raycast_head()` injects into the ultralytics namespace. Training integration via `RayCastTrainer` in Phase 10.

### GPU-bounded tests

These tests from `docs/project.md §21` require a GPU to run. All other unchecked tests are CPU-only or require only real data access.

**Phase 5-6 — Loss + Assigner:** All passed ✓
- GPU memory during assigner call ≤ 4 GB (sparse=118MB, moderate=313MB, dense=553MB)
- Mean positive assignments per GT cell: 1.82
- `L_PolarIoU` runs 10 steps without NaN
- All five loss sub-terms finite on GPU

**Phase 10 — Training Integration:** All passed ✓ (except MLflow)
- 2 GPU epochs on real PUMA data (2602 annotations, 7.3s)
- `lambda_smooth` anneals 0.05 → 0.0 over 50 epochs
- Checkpoint saved and loadable with `training_args` metadata
- Resumed training preserves RayCastDetect head and metadata
- `lambda_smooth`/`o2m_weight` logging needs custom callback (per-term losses auto-logged by Ultralytics)

**Phase 11 — Pipeline Orchestrator:** Passed ✓
- End-to-end: 202 PUMA ROIs → 202 tiles → 2 GPU epochs in 174s

**Phase 9 — ONNX/TensorRT (separate hardware):**
- TensorRT engine builds on Jetson without errors
- TensorRT FP16 output matches PyTorch FP32 within tolerance
- Jetson inference end-to-end: image in → polygon vertices out
- Throughput target: ≥ 30 tiles/sec on Jetson Orin

### Bug fixes from GPU testing

- **AMP dtype mismatch in `RayCastAssigner.select_candidates_in_gts`**: `torch.cdist` fails when `gt_xy` is float16 (AMP autocast) and `xy_centers` is float32. Fix: cast both to `.float()` before `cdist`. (`raycasted/model/tal.py:131`)
- **AMP dtype mismatch in `RayCastAssigner.get_box_metrics`**: IoU tensor is float16 under AMP but `overlaps` output is float32. Fix: `.to(overlaps.dtype)` before assignment. (`raycasted/model/tal.py:210-212`)
- **Ignore class reaching loss**: `RayCastTileDataset` did not filter `class_id=255` (Ignore) annotations, causing index-out-of-bounds when the assigner used 255 as a class index into `pd_scores` (nc=5). Fix: filter `annotations[:, 0] != 255` in `__getitem__`. (`raycasted/data/etl/loader/raycast_dataset.py:60-62`)
- **`plot_images` / `plot_predictions` crash**: Ultralytics' `plot_images()` and `plot_predictions()` expect 4-dim bboxes but get 34-dim polygon data, crashing on `xywh2xyxy()` / `xyxy2xywh()`. Override `plot_training_labels()` in `RayCastTrainer`, `plot_predictions()` and `plot_val_samples()` in `RayCastValidator` as no-ops. (`raycasted/model/train.py:149`, `raycasted/model/val.py:364-367`)
- **`nt_per_class` NoneType on epoch 1**: `RayCastDetMetrics.process()` early-return path did not set `nt_per_class` when there are no valid predictions (untrained model). Fix: compute `nt_per_class`/`nt_per_image` from `target_cls` before the early return, so GT counts are always reported regardless of prediction quality. (`raycasted/model/val.py:50-60`)
- **`cls.shape[0]` IndexError**: `.squeeze(-1)` on single-element `cls` tensor produces 0-dim scalar. Fix: `.flatten()` instead. (`raycasted/model/val.py:264`)
- **`no labels found` warning on epoch 1**: Ultralytics' `DetectionValidator.print_results()` warns when `nt_per_class.sum() == 0` (untrained model, all-zero metrics). Previously caused by early-return skipping `nt_per_class` computation — fixed above.
- **`DataLoader.reset()` AttributeError**: `RayCastTrainer.get_dataloader()` returned a plain `torch.utils.data.DataLoader`. Ultralytics' `_do_train` calls `self.train_loader.reset()` after training, which only exists on `InfiniteDataLoader`. Fix: use `ultralytics.data.build.InfiniteDataLoader` instead. (`raycasted/model/train.py:26,234`)
- **Parallel ingestion deadlock**: `ProcessPoolExecutor` with default `fork` start method deadlocks with OpenMP-backed libraries (`cv2`, `polars`). Workers hang with 0% CPU. Fix: use `multiprocessing.get_context('spawn')` for both orchestrator-level and ParquetIngestor-internal process pools. (`raycasted/data/etl/ingestors/ingestion_orchestrator.py:199`, `raycasted/data/etl/ingestors/parquet_ingestor.py:223`)
- **PanNuke parallelism**: PanNuke has only 3 registry rows (folds), so orchestrator-level parallelism uses at most 3 cores. Fix: `ParquetIngestor` now handles internal ROI-level parallelism — reads the parquet in the main process, dispatches per-ROI decode+annotate tasks to workers. Orchestrator processes Parquet rows sequentially to avoid nested process pools. Default workers: `os.cpu_count()-1`. (`raycasted/data/etl/ingestors/parquet_ingestor.py`, `raycasted/data/etl/ingestors/ingestion_orchestrator.py:152-175`)
- **Bias init in stride-space instead of normalised space**: `RayCastDetect.bias_init()` computed ray biases as `log(exp(target_px / stride) - 1)`, producing predictions ~94× too large in normalised space (softplus output ~2.17 vs target ~0.023). This caused near-zero regression and classification losses at training start. Fix: compute as `log(exp(target_px / crop_size) - 1)`. `bias_init` now accepts `crop_size` parameter; trainer re-calls it in `set_model_attributes()` with the actual `imgsz`. (`raycasted/model/head.py:155`, `raycasted/model/train.py:277`)
- **`output_image_size` / `max_size` unification**: ETL config had two independent size fields (`output_image_size` for NormalizerAndPadder, `max_size` for SpatialChunker) plus a third hardcoded `crop_size` in the trainer. Replaced with two validated Pydantic fields: `max_size` (ETL tile size) and `crop_size` (model input size). Pipeline defaults `imgsz` from config's `crop_size`. (`raycasted/data/etl/utils/config.py:36-53`)
- **Validator double normalization (mAP collapse)**: `DetectionValidator.preprocess()` divides images by 255, but `RayCastTileDataset` already returns float32 [0,1] images. Double-normalising produces ~0.004 values, causing near-zero model outputs, low-confidence predictions filtered by the 0.25 threshold, and complete mAP collapse after early epochs. Val losses were all NaN. Fix: override `preprocess` in `RayCastValidator` to skip /255. (`raycasted/model/val.py:160-173`)
- **XY decode mismatch between training and inference**: `RayCastDetectionLoss.decode_pred_xy` decoded xy as `(anchor + sigmoid) * stride`, but `RayCastDetect._inference` uses `(sigmoid * 2.0 - 0.5 + anchor) * stride`. The `* 2.0 - 0.5` expands the receptive field in inference but was missing from training, causing a systematic centroid shift. Fix: training decode now uses `(sigmoid * 2.0 - 0.5 + anchor) * stride` to match inference. (`raycasted/model/loss.py:99`)
- **Val losses persistently NaN (resolved)**: Caused by XY decode mismatch — the old training decode `(anchor + sigmoid) * stride` produced garbage xy during validation, propagating NaN through polar IoU. Resolved by the XY decode fix in `loss.py:99` which aligned training and inference decode conventions. Val losses now report correctly.
