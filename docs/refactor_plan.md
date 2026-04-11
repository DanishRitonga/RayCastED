# RayCastED Monorepo Refactor Plan

## Goal

Restructure RayCastED into a uv workspace monorepo with 4 packages, each with
isolated dependencies. This supports two deployment targets (Jetson TensorRT,
HuggingFace Spaces) and separates the heavy GPU training stack from the
CPU-only ETL pipeline.

## Current Structure

```
raycasted/
├── data/
│   ├── EDA/                    # Exploratory scripts (standalone, no refactor)
│   ├── dataset/                # Raw data gen scripts (standalone, no refactor)
│   ├── gt_generation/          # GT generation scripts (standalone, no refactor)
│   ├── utils/loader.py         # Parquet/image helpers (unused by model)
│   └── etl/
│       ├── ops/                # ← shared by model AND etl
│       │   ├── convert.py      # polygon_to_raycast, decode_to_vertices
│       │   ├── filter.py       # filter_and_clip_annotations
│       │   ├── iou.py          # polar_iou (numpy), polar_iou_torch (lazy)
│       │   ├── loss.py         # angular_smoothness_loss (numpy), _torch (lazy)
│       │   ├── augment.py      # flip/rotate permutations
│       │   └── utils.py        # count_zero_rays, validate_annotation_format
│       ├── utils/
│       │   ├── constants.py    # RAY_ANGLES, FLIP_H_IDX, format indices
│       │   └── config.py       # ETLConfig (Pydantic YAML wrapper)
│       ├── ingestors/          # 5 ingestors + orchestrator
│       ├── transform/          # SpatialChunker, NormalizerAndPadder, etc.
│       └── loader/             # RayCastTileDataset + collate
├── model/
│   ├── head.py, loss.py, tal.py, train.py, val.py, predict.py
│   ├── register.py, annotate.py
│   └── (imports from etl/ops and etl/utils/constants)
├── export/
│   ├── onnx_export.py          # (imports etl/utils/constants + model/register)
│   └── postprocess.py          # pure numpy, no raycasted imports
├── deploy/
│   ├── gradio/app.py, gradio/inference.py  # (imports model + export)
│   ├── jetson_inference.py     # (imports export/postprocess only)
│   └── build_engine.py         # CLI wrapper around trtexec
└── pipeline.py                 # (imports etl + model)
```

## Target Structure

```
RayCastED/
├── pyproject.toml                    # workspace root (no deps, just metadata)
├── packages/
│   ├── core/
│   │   ├── pyproject.toml            # numpy, shapely
│   │   └── src/raycasted_core/
│   │       ├── __init__.py
│   │       ├── constants.py          # RAY_ANGLES, FLIP_*, format indices
│   │       ├── convert.py            # polygon_to_raycast, decode_to_vertices
│   │       ├── iou.py                # polar_iou (numpy), polar_iou_torch (lazy torch)
│   │       ├── loss_ops.py           # angular_smoothness_loss (numpy + lazy torch)
│   │       ├── filter.py             # filter_and_clip_annotations
│   │       ├── augment.py            # flip/rotate permutations
│   │       └── validation.py         # count_zero_rays, validate_annotation_format
│   │
│   ├── etl/
│   │   ├── pyproject.toml            # raycasted-core, polars, opencv, pyyaml, Pillow
│   │   └── src/raycasted_etl/
│   │       ├── __init__.py
│   │       ├── config.py             # ETLConfig (Pydantic YAML wrapper)
│   │       ├── ingestors/
│   │       │   ├── _base.py
│   │       │   ├── parquet_ingestor.py
│   │       │   ├── geojson_ingestor.py
│   │       │   ├── csv_poly_ingestor.py
│   │       │   ├── mat_inst_ingestor.py
│   │       │   └── ingestion_orchestrator.py
│   │       ├── transform/
│   │       │   ├── spatialChunker.py
│   │       │   ├── normalizer.py
│   │       │   ├── stainEstimator.py
│   │       │   └── transform_orchestrator.py
│   │       └── loader/
│   │           └── raycast_dataset.py  # RayCastTileDataset
│   │
│   ├── model/
│   │   ├── pyproject.toml            # raycasted-core, torch, ultralytics
│   │   └── src/raycasted_model/
│   │       ├── __init__.py
│   │       ├── head.py               # RayCastDetect, RayRefinementBlock
│   │       ├── register.py           # register_raycast_head()
│   │       ├── loss.py               # RayCastDetectionLoss, RayCastE2ELoss
│   │       ├── tal.py                # RayCastAssigner
│   │       ├── train.py              # RayCastTrainer
│   │       ├── val.py                # RayCastValidator
│   │       ├── predict.py            # RayCastPredictor
│   │       ├── annotate.py           # RayCastAnnotator
│   │       └── collate.py            # _raycast_collate_fn (extracted from train.py)
│   │
│   └── deploy/
│       ├── pyproject.toml            # raycasted-model, gradio
│       │                             # [onnx] extras: onnx, onnxruntime, onnxsim
│       │                             # [jetson]: not in extras — tensorrt/pycuda installed via JetPack apt
│       ├── src/raycasted_deploy/
│       │   ├── __init__.py
│       │   ├── postprocess.py        # pure numpy (from export/postprocess.py)
│       │   ├── inference.py          # PyTorch inference (from deploy/gradio/inference.py)
│       │   ├── gradio_app.py         # Gradio UI (from deploy/gradio/app.py)
│       │   ├── onnx_export.py        # ONNX export (from export/onnx_export.py)
│       │   ├── jetson_runtime.py     # TensorRT runtime (from deploy/jetson_inference.py)
│       │   └── build_engine.py       # trtexec wrapper (from deploy/build_engine.py)
│       └── Dockerfile.hf             # HuggingFace Spaces container
│
├── main/                             # YAML configs, CLI entry points
│   ├── dataset.yaml
│   ├── train_pannuke.py              # thin wrapper, imports from packages
│   └── pipeline_cli.py               # replaces raycasted/pipeline.py
│
├── docs/                             # unchanged
├── tests/                            # updated import paths
└── scripts/                          # EDA scripts, gt_generation (unchanged)
```

## Dependency Graph

```
core  ←  etl
  ↑        (no dep)
core  ←  model  ←  deploy
```

- **core** depends on: numpy, shapely
- **etl** depends on: core, polars, opencv-headless, pyyaml, Pillow
- **model** depends on: core, torch, ultralytics
- **deploy** depends on: model, gradio; optional [onnx] extras, [jetson] docs

**Key property:** `etl` and `model` are fully independent. No cross-import.
Training machines install only `model` (torch stack). ETL machines install only
`etl` (CPU stack).

## Cross-Package Import Changes

### model → core (currently model → etl/ops + etl/utils)

| File | Old import | New import |
|------|-----------|------------|
| `model/loss.py` | `from raycasted.data.etl.ops.iou import polar_iou_torch` | `from raycasted_core.iou import polar_iou_torch` |
| `model/loss.py` | `from raycasted.data.etl.ops.loss import angular_smoothness_loss_torch` | `from raycasted_core.loss_ops import angular_smoothness_loss_torch` |
| `model/tal.py` | `from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch` | `from raycasted_core.iou import polar_iou_pairwise_flat_torch` |
| `model/val.py` | `from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch` | `from raycasted_core.iou import polar_iou_pairwise_flat_torch` |
| `model/annotate.py` | `from raycasted.data.etl.ops.convert import decode_to_vertices` | `from raycasted_core.convert import decode_to_vertices` |
| `model/annotate.py` | `from raycasted.data.etl.utils.constants import N_RAYS, RAY_COS, RAY_SIN` | `from raycasted_core.constants import N_RAYS, RAY_COS, RAY_SIN` |

### deploy → core

| File | Old import | New import |
|------|-----------|------------|
| `export/onnx_export.py` | `from raycasted.data.etl.utils.constants import RAY_COS, RAY_SIN` | `from raycasted_core.constants import RAY_COS, RAY_SIN` |

### model/train.py → etl/loader (THE coupling point)

Currently `train.py` directly imports `RayCastTileDataset` from
`raycasted.data.etl.loader`. This creates a model → etl dependency.

**Fix:** Move `RayCastTileDataset` and `_raycast_collate_fn` into the model
package itself. The dataset class only needs numpy and torch — no ETL deps.
The collate function is pure PyTorch. Both logically belong with the training
code, not the ETL pipeline.

The `raycast_dataset.py` file will remain in `etl/loader/` as a re-export
shim for backward compatibility during transition, but the canonical location
becomes `model/collate.py` + `model/dataset.py`.

### pipeline.py → both packages

The pipeline orchestrator wires etl + model together. It moves to
`main/pipeline_cli.py` and depends on both `raycasted-etl` and
`raycasted-model`. This is the only place both packages coexist.

## Dead Code to Remove

| Item | Reason |
|------|--------|
| `ops/iou.py`: `polar_iou()` (numpy) | Never imported anywhere |
| `ops/loss.py`: `angular_smoothness_loss()` (numpy) | Never imported anywhere |
| `data/utils/loader.py` | Standalone parquet/image helpers, unused by model or etl pipeline |

## Files NOT Moving

These stay outside the packages (no refactor needed):

- `data/EDA/` — exploratory analysis scripts, run standalone
- `data/dataset/` — raw data generation scripts
- `tests/` — only import paths change

## Files to Remove

| Item | Reason |
|------|--------|
| `data/gt_generation/` | One-off PanNuke GT script, superseded by ETL pipeline |

## Implementation Steps

### Step 1: Create workspace root

- Replace root `pyproject.toml` with workspace config
- Add `[tool.uv.workspace]` with `members = ["packages/*"]`
- Add `[tool.uv.sources]` for cross-package refs

### Step 2: Create `packages/core/`

- Move `ops/` files → `src/raycasted_core/` with new module names
- Move `utils/constants.py` → `src/raycasted_core/constants.py`
- `pyproject.toml`: numpy, shapely
- Remove dead numpy-only functions (`polar_iou`, `angular_smoothness_loss`)
- Keep lazy torch imports in `iou.py` and `loss_ops.py` — core must be
  importable without torch

### Step 3: Create `packages/etl/`

- Move ingestors, transform, config → `src/raycasted_etl/`
- `pyproject.toml`: raycasted-core, polars, opencv-headless, pyyaml, Pillow
- Fix all internal imports from `raycasted.data.etl.ops` → `raycasted_core`
- Fix all internal imports from `raycasted.data.etl.utils.constants` → `raycasted_core`

### Step 4: Create `packages/model/`

- Move all model files → `src/raycasted_model/`
- `pyproject.toml`: raycasted-core, torch, ultralytics
- Move `RayCastTileDataset` from etl/loader → model package
- Extract `_raycast_collate_fn` from train.py → `collate.py`
- Fix all imports to use `raycasted_core.*`

### Step 5: Create `packages/deploy/`

- Move deploy/ + export/ → `src/raycasted_deploy/`
- `pyproject.toml`: raycasted-model, gradio, `[onnx]` extras
- Consolidate Gradio inference into `inference.py`
- Fix imports to use `raycasted_core` and `raycasted_model`

### Step 6: Move pipeline to main/

- `pipeline.py` → `main/pipeline_cli.py`
- Depends on both `raycasted-etl` and `raycasted-model`
- CLI entry point stays the same

### Step 7: Update tests and docs

- Fix all import paths in `tests/`
- Update `CLAUDE.md` architecture section
- Update `docs/project.md` §7 repository layout

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Breaking existing training runs | Pin ultralytics version in model pyproject.toml |
| Lazy torch imports in core break | Test `import raycasted_core` in CPU-only CI |
| ETL ingestor ops tests break | Update test imports in step 7, run phase_0_5 tests |
| HuggingFace Dockerfile complexity | Deploy package has minimal deps; Dockerfile installs `raycasted-deploy[hf]` only |
| Jetson tensorrt/pycuda not pip-installable | Pre-installed via JetPack (`sudo apt install nvidia-tensorrt pycuda`). Document in deploy README, not in pyproject.toml |

## What Gets Saved (Disk)

| Package | Approx deps size | Use case |
|---------|-----------------|----------|
| core | ~50 MB | Geometry only, any machine |
| etl | ~200 MB | Data prep, CPU-only servers |
| model | ~3 GB | Training, GPU machines |
| deploy (hf) | ~3 GB + gradio | HuggingFace Spaces |
| deploy (jetson) | ~500 MB + TRT | Jetson inference |
