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
    ↓  PolygonTileDataset
[B, 3, H, W] + [M, 36] labels (normalised)
    ↓  PolygonDetect head (raycasted/model/head.py) + PolygonDetectionLoss (Phase 6)
Trained weights
    ↓  PolygonPredictor (Phase 7)
[N, 32, 2] polygon vertices  (pixel space)
```

### Key modules

**`raycasted/model/`** — Prediction head package (Phase 4). Subclasses `ultralytics.nn.modules.head.Detect` rather than modifying ultralytics in-place. Contains `PolygonDetect`, `RayRefinementBlock`, `register_polygon_head()` for namespace injection, and `PolygonDetectionLoss` stub. Files: `head.py`, `register.py`, `loss.py`.

**`raycasted/data/etl/ops/`** — Single source of truth for ALL geometry logic. Every caller imports from here; nothing is reimplemented elsewhere. PyTorch variants (`polar_iou_torch`, `angular_smoothness_loss_torch`) use **lazy imports** (`import torch` inside the function body) so this package is safe to import without PyTorch in ETL environments. `decode_pred_xy` is NOT here — it is a method of `PolygonDetectionLoss` only.

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

Normalisation (divide by `crop_size=640`) happens **only** in `PolygonTileDataset._normalise()`. Denormalisation at inference must use `crop_size` read from `model.training_args['crop_size']` — never hardcoded.

### .npz schema

Post-`TransformOrchestrator` tiles contain: `image` (uint8, HWC), `annotations` (float32, N×35), `tissue` (int32), `content_h` (int32), `content_w` (int32). Load with backward-compat key: `data.get('annotations', data.get('bboxes'))`.

### Known issues

- `MatInstIngestor._extract_raycast_annotations()` is a stub (`NotImplementedError`) — deferred, requires contour extraction from instance maps.
- Phase 0.5 deferred items (require real datasets): H&E overlay, `d_i` ≤ centroid-to-edge check, zero-ray fraction < 1%.
- Ingestor diagnostic counters not yet wired: `fallback_counter` not passed to `polygon_to_raycast()`, zero-ray logging not implemented. Must be added before first real-data run (see §12.4, NOTE-04).

### Implementation order

Follow `docs/project.md §8` strictly — each phase depends only on phases above it:

```
Phase 0   — ops/ + constants  ✓
Phase 0.5 — visual round-trip tests  ✓
Phase 1   — raycast extraction in all 3 ingestors  ✓
Phase 1.5 — IngestionOrchestrator  ✓
Phase 2   — SpatialChunker / NormalizerAndPadder / TransformOrchestrator patches  ✓
Phase 3   — PolygonTileDataset + collate_fn  ✓
Phase 4   — Model head (RayRefinementBlock, PolygonDetect, 34-dim output)  ✓
Phase 5   — PolygonAssigner (masked pairwise IoU)
Phase 6   — PolygonDetectionLoss (5 terms) + PolygonE2ELoss
Phase 7   — PolygonPredictor + PolygonAnnotator
Phase 8   — PolygonValidator
Phase 9   — ONNX export + TensorRT deployment (NVIDIA Jetson)
```

**Phase 4 approach:** Subclasses `ultralytics Detect` in `raycasted/model/head.py` instead of modifying `ultralytics/` in-place. Registration via `register_polygon_head()` injects into the ultralytics namespace. Full YAML config integration deferred to Phase 7.
