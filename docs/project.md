# RayCastED — RayCast-based End-to-end Detection
## Project Plan v4.12 — Architecture Reference & Bug Registry

> **Status:** Pre-implementation. This document is the authoritative specification.  
> **Scope:** RayCastED ETL → RayCastED training → inference → NVIDIA Jetson deployment.  
> **Dataset targets:** MoNuSAC (Parquet), PUMA (GeoJSON), PanopTILs (CSV polygons).  
> **Deployment target:** NVIDIA Jetson (Orin/Xavier) via ONNX → TensorRT.

---

## Table of Contents

1. [Change Log](#1-change-log)
2. [System Overview](#2-system-overview)
3. [Architectural Sources](#3-architectural-sources)
4. [Tech Stack](#4-tech-stack)
5. [Design Contracts (Non-Negotiable)](#5-design-contracts-non-negotiable)
6. [Data Format Contract](#6-data-format-contract)
7. [Repository Layout](#7-repository-layout)
8. [Execution Order](#8-execution-order)
9. [Module A — Prediction Head](#9-module-a--prediction-head)
10. [Module B — Loss Function](#10-module-b--loss-function)
11. [Module C — Bipartite Matcher / Assigner](#11-module-c--bipartite-matcher--assigner)
12. [Module D — ETL Pipeline](#12-module-d--etl-pipeline)
13. [Module E — DataLoader](#13-module-e--dataloader)
14. [Module F — Inference & Visualisation](#14-module-f--inference--visualisation)
15. [Module G — Validation Metrics](#15-module-g--validation-metrics)
16. [Module H — Deployment (NVIDIA Jetson)](#16-module-h--deployment-nvidia-jetson)
17. [Module I — Training Integration](#17-module-i--training-integration)
18. [Hyperparameter Reference](#18-hyperparameter-reference)
19. [Bug & Vulnerability Registry](#19-bug--vulnerability-registry)
20. [Testing Checkpoints](#20-testing-checkpoints)

---

## 1. Change Log

| Version | Changes |
|---------|---------|
| v3.0 | Initial full plan: Head, Loss, Assigner, ETL, Loader, Inference, Metrics |
| v4.0 | Identified and specified fixes for 5 vulnerabilities: VRAM explosion in pairwise IoU, E2ELoss subclass chain, target encoding mismatch, `self.no` mismatch, smoothness annealing mechanism |
| v4.1 | Markdown rewrite. Ingestion orchestrator added to scope. Per-file bug registry compiled from design review. Flip/rotate signature contract formalised. `MatInstIngestor` dispatch gap documented. |
| v4.2 | Corrected `RayCastE2ELoss.__init__` pattern. Expanded §14 with centroid-matching F1 metric and PanNuke overlap evaluation nuance. Added `representative_point()` fallback to `polygon_to_raycast`. Added mosaic/mixup disable requirement. Updated `lambda_smooth` final value guidance. |
| v4.3 | **Naming convention unified:** Switched from `star_convex` to `raycast` throughout. Leverages existing `raycast` infrastructure in `BaseDataIngestor.standardize_mpp()`. Added `content_h/w` return signature requirements for `NormalizerAndPadder`. Resolved GAP-02 (split column = `'split'`). Added annotation key standardization (`'annotations'` preferred). |
| v4.4 | Added Tech Stack section (§4) documenting all core dependencies and their usage contexts. |
| v4.5 | Added explicit filtering pipeline order for `filter_and_clip_annotations` (§12.5). Specified uniform weighting for `L_L1` loss (§10.2). Corrected architectural sources attribution: Polar-IoU from PolarMask (not LSP-DETR); LSP-DETR contributes bipartite matching and 32-ray parameterisation. Marked inference pipeline as deferred. |
| v4.6 | **Format unification:** ETL format now matches YOLO format — `[class_id, cx, cy, d_1..d_32]`. **Modular refactor:** Split `annotation_ops.py` into `ops/` folder with separate modules (convert, filter, iou, augment, loss). Moved `config.py` to `utils/`. Updated repository layout (§7). |
| v4.7 | **8 issues resolved from design review.** (1) `representative_point()` fallback now updates `cx/cy` to match the fallback point before ray casting — previously decoded polygons would be offset from their stored centroid. (2) BUG-05 VRAM fix extended with an explicit N_candidates × N_gt chunk-size guard for dense TIL fields. (3) `R_far` formula specified as `sqrt(bbox_w² + bbox_h²) × 1.1`. (4) `polar_iou_pairwise_flat` vs `polar_iou_pairwise_torch` naming resolved with explicit shape contracts per function. (5) `RayCastDetectionLoss.__init__` assigner-swap pattern made explicit. (6) `lambda_smooth` per-epoch delta and post-epoch-50 clamping behaviour stated. (7) `dedup_radius_px` formula corrected from `min(20, imgsz*0.03)` to `min(5, imgsz*0.008)`. (8) Round-trip test threshold relaxed to dataset-dependent values. |
| v4.8 | **10 issues resolved from v4.7 audit.** (1) §12.5.1 filtering diagram corrected to use unified format `[class_id, cx, cy, d_1..d_32]`. (2) §11.2 Phase 1 + Phase 2 code blocks merged into single conditional implementation. (3) §10.1 and §9.4 undefined `m` replaced with `model.model[-1]`. (4) §12.4 `_fallback_counter` specified as optional `collections.Counter` argument. (5) §14.1 ray denormalization corrected from image dimensions to `crop_size`. (6) §15 section numbering fixed (15.3 → 15.2). (7) `RayCastAssigner` constructor `eps` parameter removed (not in parent). (8) Phase 0.5 wording clarified. (9) Method name standardized to `_normalise()`. (10) §5.1 ops table synchronized with §7.1. |
| v4.9 | **5 plan errors corrected.** (1) `ANGLES` import in §12.4 code block corrected to `RAY_ANGLES`. (2) `LineString` intersection case added to §12.4 ray casting code (tangent rays). (3) §18 Phase 0 tests extended with `polar_iou_torch`, `polar_iou_pairwise_flat_torch`, and `angular_smoothness_loss_torch`. (4) §5.1 clarified: `decode_pred_xy` is a method of `RayCastDetectionLoss` only — not exported from `ops/`. (5) §10.1 Ultralytics API verification note added. |
| v4.11 | **Phase 0 implementation fixes.** (1) `polygon_to_raycast` signature corrected — returns `np.ndarray\|None` (shape 35), added `fallback_counter`, removed `use_representative_point_fallback` flag, fixed `R_far` to `sqrt(bbox_w²+bbox_h²)×1.1`. (2) `polar_iou_pairwise_torch` renamed to `polar_iou_pairwise_flat_torch`; both flat variants now accept pre-expanded `[N_cand, N_gt, 32]` inputs per §7.1 shape contract. (3) `polar_iou_pairwise` (non-flat, not in spec) removed. (4) `decode_pred_xy` removed from `ops/loss.py`. (5) `smoothness.py` naming superseded — module is `loss.py` throughout; plan updated to match. (6) `loader/__init__.py` restored (was overwritten with project.md content). |
| v4.10 | **4 specification gaps closed.** (1) §10.3 `decode_pred_xy` formula added — anchor grid decoding from grid-cell-relative to absolute normalised space. (2) §11.3 75th-percentile containment radius edge case specified — exclude zero rays; fallback to max non-zero ray when < 8 non-zero rays remain. (3) §14.1 `crop_size` source at inference specified — must be stored in model training config and read by `RayCastPredictor`. (4) §10.5 `update()` call timing clarified — lambda_smooth is still 0.001 during epoch 50's batches; reaches `smooth_end` after `update(50)` completes. |
| v4.12 | **Jetson deployment added.** New Module H (§16) specifying ONNX export and TensorRT deployment on NVIDIA Jetson. Phase 9 added to execution order. Scope updated to include deployment target. Tech stack updated with ONNX/TensorRT dependencies. Testing checkpoints added for Phase 9. |
| v4.13 | **Phase 4 implementation divergences documented.** (1) Model head implemented as `raycasted/model/` subclass package instead of in-place `ultralytics/` modification. (2) `register_raycast_head()` runtime patching for namespace injection. (3) Cell-size-aware bias initialisation (15px at 0.25 MPP). (4) `_inference()` uses YOLO decode convention `sigmoid*2-0.5`. (5) Loss stub in `raycasted/model/loss.py` (not `ultralytics/utils/loss.py`). (6) `one2many`/`one2one` confirmed as properties, not instance attributes. (7) Phase 4 testing checkpoints expanded with actual test coverage. |
| v4.14 | **Prefect dropped.** Removed Prefect from core dependencies (§4.1), dependency diagram (§4.3), and §4.5 integration section. Not justified for single-developer research project with local data — retry logic handled by simple wrapper. Also: Phase 5 `RayCastAssigner` implemented in `raycasted/model/tal.py` — subclasses `TaskAlignedAssigner` with radius containment and VRAM-safe Polar-IoU. `get_targets` NOT overridden (parent is dimension-agnostic). |
| v4.15 | **Training integration added.** New Module I (§17) specifying training pipeline that wires all model components into the Ultralytics training loop. Phase 10 added to execution order. `RayCastTrainer(DetectionTrainer)` with custom loss, data loader, and validator. YAML config integration for `RayCastDetect` head. Enables trained checkpoints for ONNX export validation (Phase 9). |

---

## 2. System Overview

Convert Ultralytics YOLOv26 (bounding-box detector) into a **multi-axis, raycast polygon detector** optimised for dense Tumour-Infiltrating Lymphocyte (TIL) detection in histopathological whole-slide images (WSIs).

**Output per detected cell:** centroid `(x_c, y_c)` + 32 radial ray distances `(d_1 … d_32)` parameterising the cell membrane along 32 equidistant angular axes.

**Pipeline stages:**

```
Raw datasets (Parquet / GeoJSON / CSV)
        ↓  IngestionOrchestrator
.npz files  [image + raycast annotations, pixel space]
        ↓  TransformOrchestrator  (SpatialChunker + NormalizerAndPadder)
.npz tiles  [content_h, content_w preserved]
        ↓  RayCastTileDataset
[B, 3, H, W] + [M, 36] labels  (normalised)
        ↓  RayCastED  (RayCastE2ELoss + RayCastAssigner)
Trained weights
        ↓  RayCastPredictor
[N, 32, 2] polygon vertices  (pixel space)
```

---

## 3. Architectural Sources

| Source | Contribution |
|--------|-------------|
| **YOLOv26** | NMS-free end-to-end Hungarian matching, convolutional backbone, `E2ELoss` dual-assignment framework |
| **LSP-DETR** | End-to-end bipartite matching for polar predictions, 32-ray parameterisation |
| **PolarMask** | Polar-IoU loss formulation — efficient differentiable IoU approximation for ray-based detection in CNNs |
| **StarDist** | Angular spacing convention (θ₁ = 0° → East, counter-clockwise), center-of-mass anchoring |
| **CPP-Net** | `RayRefinementBlock`: 3×3 depthwise conv before final projection for neighbour-blended ray prediction |
| **SplineDist** | Angular smoothness regularisation — circular first-difference penalty on predicted rays |
| **RayCastED ETL** | WSI tiling (`SpatialChunker`), stain normalisation (Macenko), white padding, `.npz` cache, existing `raycast` annotation type infrastructure |

---

## 4. Tech Stack

### 4.1 Core Dependencies

| Library | Version | Purpose | Usage Context |
|---------|---------|---------|---------------|
| **PyTorch** | ≥2.0 | Deep learning framework | Model definition, training loop, loss computation, inference |
| **Ultralytics** | ≥8.3 | YOLO framework | YOLOv26 backbone, training infrastructure, detection utilities |
| **NumPy** | ≥1.24 | Numerical computing | ETL annotation processing, geometry operations, `.npz` I/O |
| **Polars** | ≥0.20 | DataFrame operations | File registry management, CSV parsing, split assignment |
| **SciPy** | ≥1.11 | Scientific computing | `.mat` file loading (`loadmat`), `ndimage.find_objects` for instance masks |
| **OpenCV** | ≥4.8 | Image processing | Image I/O, contour extraction, resize, padding, visualisation |
| **Shapely** | ≥2.0 | Geometry operations | Polygon manipulation, ray-casting, IoU computation |
| **Pydantic** | ≥2.0 | Data validation | Configuration validation, schema enforcement |
| **pathlib** | stdlib | File path handling | Cross-platform path operations, directory structure management |

### 4.2 Secondary Dependencies

| Library | Purpose |
|---------|---------|
| **PyYAML** | Configuration file parsing |
| **orjson** | Fast JSON parsing for GeoJSON files |
| **MLflow** | Experiment tracking, model versioning, metric logging |

### 4.2.1 Deployment Dependencies (Jetson)

| Library | Version | Purpose | Where installed |
|---------|---------|---------|-----------------|
| **onnx** | ≥1.14 | ONNX model format | Training machine (export) |
| **onnxruntime** | ≥1.16 | ONNX validation & CPU inference | Training machine (validation) |
| **onnx-simplifier** | ≥0.4 | Graph optimisation before TensorRT conversion | Training machine (export) |
| **TensorRT** | ≥8.6 | Optimised inference engine | Jetson device only |
| **pycuda** / **cuda-python** | — | TensorRT host↔device memory transfers | Jetson device only |

### 4.3 Dependency by Pipeline Stage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION STAGE                                    │
│  polars • numpy • opencv • scipy • pathlib • shapely • orjson • pyyaml     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRANSFORM STAGE                                    │
│  numpy • opencv • pathlib • shapely • scipy (stain estimation)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING STAGE                                     │
│  pytorch • ultralytics • numpy • shapely (metrics)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE STAGE                                    │
│  pytorch • numpy • opencv • shapely                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXPORT (Training Machine)                               │
│  pytorch • onnx • onnxruntime • onnx-simplifier                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT (NVIDIA Jetson)                              │
│  tensorrt • pycuda • numpy • opencv                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Environment Isolation

The ETL pipeline is designed to run **without PyTorch installed**. This is enforced by:

1. **Lazy imports in `ops/`:** PyTorch-specific functions in `ops/iou.py` and `ops/loss.py` use `import torch` inside function bodies, allowing the modules to be imported in ETL-only environments.

2. **Separate requirement files:**
   ```
   requirements-etl.txt      # ETL-only dependencies
   requirements-train.txt    # Training dependencies (includes PyTorch, Ultralytics)
   requirements-full.txt     # Combined dependencies
   ```

3. **Docker images:**
   ```
   raycasted-etl:latest        # Lightweight ETL-only image
   raycasted-train:latest      # Full training image with GPU support
   ```

---

## 5. Design Contracts (Non-Negotiable)

### 5.1 Single Source of Truth for Geometry

Every piece of polygon/raycast geometry logic lives in **`raycasted/data/etl/ops/`**.  
The modules are **imported** by every caller. They are **never duplicated**.

**Module responsibilities:**

| Module | Functions | Purpose |
|--------|-----------|---------|
| `ops/convert.py` | `polygon_to_raycast`, `decode_to_vertices` | Polygon ↔ ray conversion |
| `ops/filter.py` | `filter_and_clip_annotations` | Crop-region filtering |
| `ops/iou.py` | `polar_iou`, `polar_iou_pairwise_flat`, `polar_iou_torch`, `polar_iou_pairwise_flat_torch` | IoU computation |
| `ops/augment.py` | `flip_horizontal`, `flip_vertical`, `rotate_90` | Geometric augmentations |
| `ops/loss.py` | `angular_smoothness_loss`, `angular_smoothness_loss_torch` | Regularization |

> **`decode_pred_xy` is NOT in `ops/`.** It is a method of `RayCastDetectionLoss` only (§10.3). It requires knowledge of the anchor grid layout and has no meaning outside the training loss context. Placing it in `ops/` would violate the ETL/training isolation contract — `ops/` must be importable without PyTorch.

**Callers:**
- `raycasted/data/etl/ingestors/*.py` (NumPy, offline ETL)
- `raycasted/data/etl/transform/spatialChunker.py` (NumPy, offline ETL)
- `raycasted/data/etl/loader/raycast_dataset.py` (NumPy, online DataLoader)
- `raycasted/model/loss.py` (PyTorch, training)
- `raycasted/model/tal.py` (PyTorch, assignment)
- `raycasted/model/predict.py` (PyTorch, inference)

**PyTorch variants** (`polar_iou_torch`, `angular_smoothness_loss_torch`, etc.) live in `ops/iou.py` and `ops/loss.py` with **lazy imports** (`import torch` inside the function body) so the ops modules are safe to import in ETL environments without PyTorch installed.

### 5.2 Coordinate Space is Always Explicit

| Stage | Coordinate space | Who converts |
|-------|-----------------|--------------|
| Ingestors → `.npz` | Pixel | `standardize_mpp` in `BaseDataIngestor` |
| SpatialChunker → `.npz` tiles | Pixel | `filter_and_clip_annotations` |
| DataLoader emission | Normalised `[0, 1]` | `RayCastTileDataset._normalise()` — **only place normalisation happens** |
| Loss centroid computation | Normalised `[0, 1]` | `RayCastDetectionLoss.decode_pred_xy()` — **only place xy prediction decoding happens** |
| Inference output | Pixel | `decode_to_vertices()` — **only place denormalisation happens** |

No function silently converts between spaces. Every function signature that accepts spatial coordinates must document which space it expects.

### 5.3 Class Responsibility Boundaries

| Class | Owns | Does NOT own |
|-------|------|-------------|
| `IngestionOrchestrator` | Config-driven ingestor dispatch, `.npz` writing | Tiling, normalisation |
| `SpatialChunker` | Offline tiling geometry | Normalisation, tensors |
| `NormalizerAndPadder` | Stain normalisation, padding, `content_h/w` output | Coordinate transforms |
| `TransformOrchestrator` | Tiling + normalisation pipeline | Ingestion, model |
| `AnnotationOps` | All polygon/ray geometry math (via `ops/` modules) | I/O, image processing |
| `RayCastTileDataset` | Online augmentation, tensor emission | Offline ETL, model |
| `RayCastDetectionLoss` | Loss computation | Matching, assignment |
| `RayCastAssigner` | Hungarian cost matrix | Loss terms |
| `RayCastE2ELoss` | Dual-assignment orchestration + weight decay + smoothness annealing | Individual loss terms |
| `RayCastPredictor` | Inference decode | Training |
| `RayCastAnnotator` | Visualisation | Metrics |
| `RayCastValidator` | Metric computation | Visualisation |

### 5.4 Angular Convention (Immutable)

```
θ_1 = 0° → East (+X axis)
Angles increase counter-clockwise.
Angular spacing: 11.25° (= 2π / 32)
```

Defined once in `raycasted/data/etl/utils/constants.py`. Imported everywhere. Never recomputed inline. Any label generation script that uses a different convention will produce silently wrong training data.

---

## 6. Data Format Contract

### 6.1 Unified Format (ETL + YOLO)

All stages use the **same format** — no conversion needed between ETL and model:

```
[class_id, cx, cy, d_1, ..., d_32]
     0        1     2     3         34
shape: (N, 35), dtype: float32
class_id at index 0
```

| Stage | Coordinate space | Note |
|-------|-----------------|------|
| Ingestors → `.npz` | Pixel | `cx`, `cy`, `d_i` in pixel units |
| DataLoader emission | Normalised `[0, 1]` | Divide by `crop_size` |
| Loss / Assigner | Normalised `[0, 1]` | Same as DataLoader output |
| Inference output | Pixel | Denormalise by tile dimensions |

**Benefits of unified format:**
- No index reshuffling between ETL and model
- Consistent with Ultralytics/YOLO conventions
- Fewer opportunities for silent bugs

### 6.2 Collated Batch Format

```
[batch_idx, class_id, cx_norm, cy_norm, d_1_norm, ..., d_32_norm]
      0          1        2        3         4              35
shape: (sum_M, 36), dtype: float32
```

### 6.3 `.npz` Schema (post-TransformOrchestrator)

```
image:       uint8  [H, W, 3]     — padded to tile_size × tile_size
annotations: float32 [N, 35]     — unified format, pixel space
tissue:      int32  scalar
content_h:   int32  scalar        — real tissue height before white padding
content_w:   int32  scalar        — real tissue width before white padding
```

`content_h/w` is critical: constrains random crop origins to tissue-only regions and provides the crop boundary for ray clipping.

**Key naming convention:** All `.npz` files use `'annotations'` as the key for annotation arrays. The `TransformOrchestrator` reads with backward compatibility:
```python
annotations = data.get('annotations', data.get('bboxes'))
```

---

## 7. Repository Layout

```
raycasted/
├── data/
│   ├── etl/
│   │   ├── ingestors/
│   │   │   ├── _base.py                 EXISTS — BaseDataIngestor (add raycast handling)
│   │   │   ├── ingestion_orchestrator.py  EXISTS — drives ingestors from YAML config
│   │   │   ├── geojson_ingestor.py      MODIFY — implement _extract_raycast_annotations()
│   │   │   ├── csv_poly_ingestor.py     MODIFY — implement _extract_raycast_annotations()
│   │   │   ├── parquet_ingestor.py      MODIFY — implement _extract_raycast_annotations()
│   │   │   └── mat_inst_ingestor.py     EXISTS — method 3 (no raycast needed yet)
│   │   └── transform/
│   │       ├── spatialChunker.py        MODIFY — add _slice_raycast()
│   │       ├── normalizer.py            MODIFY — return content_h, content_w
│   │       ├── stainEstimator.py        EXISTS — no changes needed
│   │       └── transform_orchestrator.py  MODIFY — save content dims to .npz, use 'annotations' key
│   ├── ops/                             NEW — modular geometry operations
│   │   ├── __init__.py                  NEW — re-exports all ops
│   │   ├── convert.py                   NEW — polygon_to_raycast, raycast_to_annotation, decode_to_vertices
│   │   ├── filter.py                    NEW — filter_and_clip_annotations
│   │   ├── iou.py                       NEW — polar_iou, polar_iou_pairwise_flat, torch variants
│   │   ├── augment.py                   NEW — flip_horizontal, flip_vertical, rotate_90
│   │   └── loss.py                      NEW — angular_smoothness_loss
│   ├── utils/
│   │   ├── __init__.py                  NEW
│   │   ├── config.py                    MOVED from etl/ — Pydantic ETL config
│   │   └── constants.py                 NEW — angular constants, permutation indices, format indices
│   └── loader/
│       └── raycast_dataset.py           NEW — RayCastTileDataset, collate_fn
├── model/                               NEW — prediction head (subclass, not in-place ultralytics mod)
│   ├── __init__.py                      NEW — re-exports RayCastDetect, RayRefinementBlock, register
│   ├── head.py                          NEW — RayRefinementBlock + RayCastDetect(Detect)
│   ├── register.py                      NEW — register_raycast_head() namespace injection
│   ├── loss.py                          NEW — RayCastDetectionLoss stub (full impl in Phase 6)
│   └── tal.py                           NEW — RayCastAssigner(TaskAlignedAssigner)

ultralytics/                              # NOT modified in-place — subclass approach used
├── nn/modules/
│   └── head.py                          Referenced — Detect base class (no modification)
├── utils/
│   ├── loss.py                          Referenced — v8DetectionLoss base class (no modification)
│   ├── tal.py                           Referenced — TaskAlignedAssigner base class (no modification)
│   └── metrics.py                       Referenced — metric utilities (no modification)
└── models/yolo/detect/
    ├── predict.py                        Referenced — predictor base class (no modification)
    ├── val.py                            Referenced — validator base class (no modification)
    ├── train.py                          Referenced — training loop (no modification)
    └── plotting.py                       Referenced — annotator base class (no modification)
```

### 7.1 Ops Module Structure

The `ops/` folder provides a clean, modular API for all geometry operations:

| Module | Functions | Primary Caller |
|--------|-----------|----------------|
| `convert.py` | `polygon_to_raycast`, `raycast_to_annotation`, `decode_to_vertices`, `raycast_to_polygon` | Ingestors, Inference |
| `filter.py` | `filter_and_clip_annotations` | SpatialChunker, DataLoader |
| `iou.py` | `polar_iou`, `polar_iou_pairwise_flat`, `polar_iou_torch`, `polar_iou_pairwise_flat_torch` | Loss, Assigner |
| `augment.py` | `flip_horizontal`, `flip_vertical`, `rotate_90` | DataLoader |
| `loss.py` | `angular_smoothness_loss`, `angular_smoothness_loss_torch` | Loss |
| `utils.py` | `count_zero_rays`, `validate_annotation_format` | Ingestors, diagnostics |

**Function shape contracts for `iou.py` (naming is now fixed):**

| Function | Input shape | Output shape | Used by |
|----------|-------------|--------------|---------|
| `polar_iou` | `[N, 32]`, `[N, 32]` (NumPy) | `[N]` | ETL diagnostics |
| `polar_iou_torch` | `[N, 32]`, `[N, 32]` (Tensor) | `[N]` | `RayCastDetectionLoss` |
| `polar_iou_pairwise_flat` | `[N_cand, N_gt, 32]`, `[N_cand, N_gt, 32]` (NumPy) | `[N_cand, N_gt]` | — |
| `polar_iou_pairwise_flat_torch` | `[N_cand, N_gt, 32]`, `[N_cand, N_gt, 32]` (Tensor) | `[N_cand, N_gt]` | `RayCastAssigner` |

**Naming rationale:** The `_flat` suffix means the batch dimension has already been collapsed by the caller's per-batch loop. It accepts a flat `[N_cand, N_gt, 32]` slice, not the full `[B, N_anchors, N_gt, 32]` expansion. There is no full-batch pairwise function — that tensor is what causes BUG-05.

**Import convention:**
```python
# Preferred: import from ops package
from raycasted.data.etl.ops import polygon_to_raycast, filter_and_clip_annotations
from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch

# Or import specific module
from raycasted.data.etl.ops.convert import polygon_to_raycast
```

---

## 8. Execution Order

Implement strictly in this order. Each phase depends only on phases above it.

```
Phase 0   — constants.py + ops/ modules (convert, filter, iou, augment, loss, utils)
Phase 0.5 — Unit-test `polygon_to_raycast` on manually constructed sample polygons; decode to vertices and render onto sample H&E crops to verify angular convention and boundary alignment before committing to full ETL ingestion
Phase 1   — raycast implementation in GeoJSON, CSV, Parquet ingestors
Phase 1.5 — IngestionOrchestrator
Phase 2   — SpatialChunker, NormalizerAndPadder, TransformOrchestrator patches
Phase 3   — RayCastTileDataset + collate_fn
Phase 4   — Model Head (RayRefinementBlock, 34-dim output)
Phase 5   — RayCastAssigner (masked pairwise IoU)
Phase 6   — RayCastDetectionLoss + RayCastE2ELoss
Phase 7   — RayCastPredictor + RayCastAnnotator
Phase 8   — RayCastValidator
Phase 9   — ONNX export + TensorRT deployment (NVIDIA Jetson)
Phase 10  — Training integration (RayCastTrainer + YAML config + MLflow)
```

---

## 9. Module A — Prediction Head

**Files:**
- `raycasted/model/head.py` — `RayRefinementBlock`, `RayCastDetect(Detect)` (subclass approach)
- `raycasted/model/register.py` — `register_raycast_head()` namespace injection
- `raycasted/model/__init__.py` — package re-exports

**Implementation approach (diverges from v4.0–v4.12 plan):** The original plan specified in-place modification of `ultralytics/nn/modules/head.py`. The actual implementation subclasses `Detect` in a separate `raycasted/model/` package. This avoids forking ultralytics and keeps all RayCastED-specific code under `raycasted/`. Runtime registration via `register_raycast_head()` injects `RayCastDetect` into the ultralytics namespace so YAML configs can resolve it (full YAML integration deferred to Phase 7).

**Ultralytics API notes (verified against installed version ≥8.3):**
- `one2many` and `one2one` are **properties** returning `dict(box_head=self.cv2, cls_head=self.cv3)` — not instance attributes. Replacing `self.cv2` in `RayCastDetect.__init__` automatically updates what `one2many['box_head']` returns.
- `parse_model()`'s module frozenset is a **local variable** — cannot be patched externally. Full YAML integration requires either monkey-patching `parse_model` or manual model construction (deferred to Phase 7).

### 9.1 RayRefinementBlock

3×3 depthwise conv + GroupNorm + SiLU + residual skip.  
Placed before the final `Conv2d(c2, 34, 1)` projection.  
Blends features from spatially neighbouring anchor points before committing to ray predictions — reduces jagged outputs without Transformer overhead (CPP-Net design).

- `num_groups=8` for GroupNorm; reduce to 4 if `channels < 64`.

### 9.2 Regression Branch

Output: `[B, N_anchors, 34]` = `[xy_offset(2), rays(32)]`.  
**DFL is removed.** DFL is specific to `(x,y,w,h)` box regression and has no meaning for polar coordinates.

Replace the existing `cv2` regression conv stack with:
```
Conv(x, c2, 3) → Conv(c2, c2, 3) → RayRefinementBlock(c2) → Conv2d(c2, 34, 1)
```

### 9.3 Output Activations

- Channels 0–1 (xy): **Sigmoid** → offset from anchor grid cell, bounded `[0, 1]`
- Channels 2–33 (rays): **Softplus** → strictly positive, smooth near zero

**Why Softplus over ReLU:** ReLU can output exact zero, causing `max(d_pred, d_gt) = 0` → division-by-zero in Polar-IoU denominator. Softplus is always strictly positive and has smooth gradients near zero.

### 9.4 `self.no` Override

In `RayCastDetectionLoss.__init__`, explicitly override:
```python
self.no = model.model[-1].nc + 34   # parent sets: m.nc + m.reg_max * 4  ← wrong for polygon head
self.use_dfl = False
```
Any downstream code that reads `self.no` to slice prediction tensors will silently produce wrong shapes if this is not overridden. See BUG-02.

### 9.5 Inference Decode Convention

`RayCastDetect._inference()` decodes xy offsets using the YOLO convention:
```python
xy_abs = (xy_offset.sigmoid() * 2.0 - 0.5 + anchor_grid) * stride
```

The `* 2.0 - 0.5` transform maps the Sigmoid output from `[0, 1]` to `[-0.5, 1.5]` relative to the anchor grid cell, allowing predictions to extend slightly beyond cell boundaries — matching the standard YOLO decode behaviour inherited from `Detect`.

This differs from the training-time `decode_pred_xy()` formula in §10.3 which operates in normalised space for loss computation. The two decodes serve different purposes and intentionally use different formulas.

### 9.6 Bias Initialisation — Cell-Size-Aware

Ray biases are initialised to produce ~15px rays at inference, calibrated for typical lymphocyte size at 0.25 MPP resolution (7–10 μm diameter → 28–40px → radius ~15px):

```python
target_ray_px = 15.0  # reasonable lymphocyte radius at 0.25 MPP
bias[:2] = 2.0  # xy: sigmoid(2.0) ≈ 0.88
bias[2:] = log(exp(target_ray_px / stride) - 1)  # inverse softplus → ~15px
```

Per-scale bias values:
| Scale | Stride | Ray bias | Decoded ray |
|-------|--------|----------|-------------|
| P3 | 8 | 1.709 | 15.0px |
| P4 | 16 | 0.441 | 15.0px |
| P5 | 32 | -0.514 | 15.0px |

XY biases remain at 2.0 (sigmoid → ~0.88, encouraging initial detections).

---

## 10. Module B — Loss Function

**Files:**
- `raycasted/model/loss.py` — `RayCastDetectionLoss(v8DetectionLoss)` stub (Phase 4)
- Full 5-term loss implementation deferred to Phase 6 (may remain in `raycasted/model/` or move alongside ultralytics integration)

> **Note:** The original plan specified `ultralytics/utils/loss.py`. The Phase 4 stub lives in `raycasted/model/loss.py` alongside the head, following the subclass package approach. The full Phase 6 implementation will determine whether it stays here or requires deeper ultralytics integration.

### 10.1 Subclass Chain

```
v8DetectionLoss
    └── RayCastDetectionLoss        replaces bbox/DFL logic with 5 polygon terms
E2ELoss
    └── RayCastE2ELoss              wires RayCastDetectionLoss into both branches
                                    owns lambda_smooth annealing via update()
```

> **Ultralytics API verification required before implementing §10–§11.** Before writing any model-side code, confirm these three things against the actual installed Ultralytics version:
> 1. `E2ELoss.__init__` accepts `loss_fn` as a keyword argument (search `tal.py` / `loss.py` for `E2ELoss`).
> 2. `v8DetectionLoss.__init__` stores `self.topk`, `self.alpha`, `self.beta` as instance attributes — not as local variables — so they are available after `super().__init__()` when constructing `RayCastAssigner`.
> 3. The `TaskAlignedAssigner` constructor signature matches what `RayCastAssigner` inherits from.
> If any of these differ, adapt the subclass patterns below before proceeding.

**`RayCastE2ELoss` initialisation:** The current Ultralytics `E2ELoss.__init__` accepts `loss_fn` as a default-argument parameter (`loss_fn=v8DetectionLoss`) and uses it to instantiate both branches. `RayCastE2ELoss` must override `__init__` to pass `RayCastDetectionLoss` explicitly rather than relying on the default:

```python
class RayCastE2ELoss(E2ELoss):
    def __init__(self, model):
        super().__init__(model, loss_fn=RayCastDetectionLoss)
        self.smooth_start         = 0.05
        self.smooth_end           = 0.0    # configurable floor; set to 0.01 if shapes become jagged after epoch 50
        self.smooth_anneal_epochs = 50
```

Calling `super().__init__` with the correct `loss_fn` is the preferred pattern because it inherits the full decay-parameter initialisation (`o2m`, `o2o`, `o2m_copy`, `final_o2m`) without duplicating it.

**`RayCastDetectionLoss` assigner swap — explicit pattern:**
`v8DetectionLoss.__init__` instantiates `TaskAlignedAssigner` and stores it as `self.assigner`. `RayCastDetectionLoss.__init__` must call `super().__init__()` first to inherit all setup, then **immediately replace** the assigner on the next line:

```python
class RayCastDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)              # inherits self.assigner = TaskAlignedAssigner(...)
        self.assigner = RayCastAssigner(     # replaces it — one line, right after super()
            num_classes=self.nc,
            topk=self.topk,
            alpha=self.alpha,
            beta=self.beta,
        )
        self.no       = model.model[-1].nc + 34  # override IMMEDIATELY — see BUG-02
        self.use_dfl  = False
```

This pattern is safe because:
- `super().__init__()` builds the full inherited state including the wrong assigner
- The one-line replacement overwrites `self.assigner` before any forward pass can use it
- All other inherited attributes (`nc`, `topk`, `alpha`, `beta`) are valid by the time `RayCastAssigner` is constructed because they were set by `super().__init__()`

If `super().__init__` is skipped and both branches are instantiated manually, all future changes to `E2ELoss.__init__` (new decay fields, new loss branches) must be duplicated here by hand. That is a maintenance trap.

### 10.2 Total Loss

```
L = λ_cls × L_cls  +  λ_xy × L_xy  +  λ_L1 × L_L1  +  λ_piou × L_PolarIoU  +  λ_smooth × L_smooth
```

| Term | Formula | λ initial | Annealed |
|------|---------|-----------|---------|
| `L_cls` | BCE/Focal | 0.5 | No |
| `L_xy` | Huber β=0.01 on decoded centroid | 1.0 | No |
| `L_L1` | Uniform-weighted MAE on 32 rays | 1.0 | No |
| `L_PolarIoU` | `1 − PolarIoU(d_pred, d_gt)` | 2.0 | No |
| `L_smooth` | Circular first-difference on d_pred | 0.05 | → 0 at epoch 50 |

**`L_L1` weighting scheme:** Uniform weighting (`w_i = 1/32` for all 32 rays). Rationale: TIL shape accuracy matters equally in all directions; no angular bias is desired. The formula is:

```
L_L1 = (1/32) × Σ|d_pred_i - d_gt_i|    for i in 1..32
```

All five sub-losses must be logged separately to MLflow.

### 10.3 `decode_pred_xy` — Required New Method

The head outputs grid-cell-relative Sigmoid offsets in `[0, 1]`. GT centroids from the DataLoader are absolute normalised coordinates in `[0, 1]`. These are in different spaces and cannot be compared directly.

`RayCastDetectionLoss` must implement `decode_pred_xy()` to convert Sigmoid xy outputs to absolute normalised space **before** computing `L_xy`. This mirrors `v8DetectionLoss.bbox_decode()` in purpose.

**Decoding formula:**

```python
def decode_pred_xy(self, xy_raw: Tensor, anchor_grid: Tensor, stride: Tensor) -> Tensor:
    """
    Decode grid-cell-relative Sigmoid offsets to absolute normalised coordinates.

    Args:
        xy_raw:      [B, N_anchors, 2] — raw head output (pre-activation)
        anchor_grid: [N_anchors, 2]    — grid cell top-left positions in pixel space
        stride:      [N_anchors, 1]    — feature map stride for each anchor

    Returns:
        xy_abs: [B, N_anchors, 2] — absolute normalised coords in [0, 1]
    """
    # sigmoid output is cell-relative offset in [0, 1]
    xy_offset = xy_raw.sigmoid()
    # add anchor pixel position, then normalise by crop_size (training tile size)
    xy_abs = (anchor_grid + xy_offset * stride) / self.imgsz
    return xy_abs
```

`anchor_grid` and `stride` are produced by the same anchor generation utility that `v8DetectionLoss` uses — call the parent's `make_anchors()` (or equivalent) and pass the result through.

The **inverse** — encoding GT centroids into grid-cell-relative space — must NOT be done. It creates a fragile coupling between `get_targets()` and the anchor grid layout. See BUG-03.

### 10.4 `preprocess` Override

`v8DetectionLoss.preprocess()` calls `xywh2xyxy` and scales by `imgsz[[1,0,1,0]]`. Both are wrong for 34-dim polar targets and must not be inherited.

`RayCastDetectionLoss.preprocess()` must:
- Accept `(N, 36)` batch targets: `[batch_idx, class, cx_norm, cy_norm, d_1..d_32_norm]`
- Apply no additional scaling — all spatial quantities arrive normalised from the DataLoader
- Return `[B, N_gt_max, 35]`

See BUG-04.

### 10.5 Smoothness Annealing Mechanism

`RayCastE2ELoss.update()` is called once per epoch by the training loop. It must:
1. Call `super().update()` — handles inherited `o2m`/`o2o` decay
2. Linearly anneal `lambda_smooth` on **both** `self.one2many` and `self.one2one` loss instances

**Per-epoch delta and clamping behaviour:**

```python
def update(self, epoch: int):
    super().update(epoch)   # handles o2m/o2o decay — do not skip

    # Per-epoch delta: start=0.05, end=0.0 (or smooth_end), over 50 epochs
    # delta = (0.05 - 0.0) / 50 = 0.001 per epoch
    delta = (self.smooth_start - self.smooth_end) / self.smooth_anneal_epochs

    new_lambda = max(
        self.smooth_end,
        self.smooth_start - delta * epoch,
    )

    # Apply to both loss branches
    self.one2many.lambda_smooth = new_lambda
    self.one2one.lambda_smooth  = new_lambda

    # Log to MLflow every epoch — required for interaction monitoring (see GAP-05)
    mlflow.log_metrics({
        'train/lambda_smooth': new_lambda,
        'train/o2m_weight':    self.o2m,
    }, step=epoch)
```

**Call timing:** `update(epoch)` is called by the training loop **after all batches for that epoch complete** (standard Ultralytics pattern). Therefore:
- During all batches of epoch 0: `lambda_smooth = 0.05` (the `__init__` value — `update` has not yet run)
- After epoch 49's batches, `update(49)` fires: `lambda_smooth = max(0.0, 0.05 - 0.001×49) = 0.001`
- After epoch 50's batches, `update(50)` fires: `lambda_smooth = max(0.0, 0.05 - 0.001×50) = 0.0`

So `lambda_smooth` is still **0.001 during epoch 50's batches** and only reaches `smooth_end` after `update(50)` completes. The test checkpoint assertion `lambda_smooth = 0.0 at epoch 50` means: after `update(50)` is called, not during epoch 50's training step.

**After epoch 50:** `lambda_smooth` is clamped at `smooth_end` (default `0.0`) for all remaining training epochs. `max(self.smooth_end, ...)` enforces this — the value never goes below the floor. No conditional branch is needed.

**`smooth_end` configuration:** If shapes become visually jagged after epoch 50, the `RayRefinementBlock` alone is insufficient to enforce smoothness. Set `smooth_end=0.01` in the config and retrain. This is a one-line config change, not a code change.

**Interaction risk (GAP-05):** The inherited `o2m` decay and `lambda_smooth` annealing are both active from epoch 0 to 50. `delta = 0.001` per epoch is small in absolute terms, but MLflow must log both scalars from epoch 1 so the interaction is observable. If training instability appears in epochs 10–30, delay the annealing start by setting `smooth_start_epoch=20` in the config and offsetting the delta calculation:

```python
effective_epoch = max(0, epoch - self.smooth_start_epoch)
new_lambda = max(self.smooth_end, self.smooth_start - delta * effective_epoch)
```

---

## 11. Module C — Bipartite Matcher / Assigner

**File:** `raycasted/model/tal.py` — `RayCastAssigner(TaskAlignedAssigner)` (implemented in Phase 5)

> **Note:** Following the subclass approach established in Phase 4, `RayCastAssigner` subclasses `TaskAlignedAssigner` in a separate file under `raycasted/model/`, not by modifying `ultralytics/utils/tal.py`.

### 11.1 `RayCastAssigner(TaskAlignedAssigner)`

Two methods are overridden. A third (`get_targets`) is intentionally **not** overridden — the parent implementation is dimension-agnostic (uses `gt_bboxes.shape[-1]` dynamically) and produces 34-dim polygon targets correctly without modification.

| Method | Override reason |
|--------|----------------|
| `get_box_metrics()` | Replace box IoU with Polar-IoU; apply VRAM-safe masked expansion (BUG-05) |
| `select_candidates_in_gts()` | Replace box-containment check with polar-radius containment |
| `get_targets()` | **Not overridden** — parent is dimension-agnostic, works for 34-dim targets |

See BUG-06.

### 11.2 VRAM-Safe Masked Polar-IoU

**Problem:** Naive `[B, N_anchors, N_gt, 32]` expansion occupies ~8.6 GB at standard training configuration (B=16, N_anchors=8400, N_gt=500). This causes an immediate OOM crash.

**Unified fix: Batch loop + spatial mask + conditional chunking**

Apply `select_candidates_in_gts()` spatial mask first (uses only xy, no ray expansion). Loop over the batch dimension, computing pairwise IoU only for the subset of anchors that pass the spatial filter per image. For dense TIL fields, conditionally chunk the computation to stay under memory budget:

```python
MAX_FLAT_PAIRS = 1_000_000   # 1M pairs × 32 rays × 4 bytes = 128 MB per chunk
iou_matrix = torch.zeros(B, N_anchors, N_gt, device=device)

for b in range(B):
    # candidate_mask: [N_anchors] boolean — from select_candidates_in_gts
    cand_idx = candidate_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [N_cand]
    N_cand   = cand_idx.shape[0]

    if N_cand == 0:
        continue

    pd_d_cand = pd_rays[b, cand_idx, 2:]          # [N_cand, 32]
    gt_d_b    = gt_rays[b, :, 2:]                 # [N_gt, 32]

    if N_cand * N_gt <= MAX_FLAT_PAIRS:
        # Small enough: compute in one shot
        pd_exp = pd_d_cand[:, None, :].expand(-1, N_gt, -1)
        gt_exp = gt_d_b[None, :, :].expand(N_cand, -1, -1)
        iou_matrix[b, cand_idx] = polar_iou_pairwise_flat_torch(pd_exp, gt_exp)
    else:
        # Dense scene: chunk to stay under memory budget
        CHUNK_SIZE = max(1, MAX_FLAT_PAIRS // N_gt)
        for chunk_start in range(0, N_cand, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, N_cand)
            chunk_idx = cand_idx[chunk_start:chunk_end]

            pd_chunk = pd_d_cand[chunk_start:chunk_end][:, None, :].expand(-1, N_gt, -1)
            gt_chunk = gt_d_b[None, :, :].expand(chunk_idx.shape[0], -1, -1)

            iou_matrix[b, chunk_idx] = polar_iou_pairwise_flat_torch(pd_chunk, gt_chunk)
```

**Memory budget verification:** Profile GPU memory on the first training batch in three regimes: sparse (< 50 cells/image), moderate (~200 cells/image), dense (> 400 cells/image). The assigner call must not exceed 4 GB in any regime. `MAX_FLAT_PAIRS` can be tuned upward if the GPU has headroom.

**Expected N_candidates in practice:**
- Sparse scene (50 cells): ~200–400 candidates (4–8 per cell at radius_scale=1.5)
- Dense TIL field (400 cells): up to ~5000 candidates; chunking guard triggers when N_cand × N_gt > 1M

### 11.3 Candidate Containment — 75th-Percentile Radius

Use the 75th-percentile ray value (not `mean`) as the containment radius estimate. More robust for elongated or dented cells; less noisy than max.

**Zero-ray handling:** Compute the percentile over **non-zero rays only**. Zero rays indicate geometry failures in `polygon_to_raycast` and must not deflate the radius estimate. If fewer than 8 non-zero rays remain (< 25% of 32), fall back to the maximum non-zero ray value instead. If all rays are zero, the GT cell is degenerate — skip it in the assignment cost matrix (set its column to 0.0 in `iou_matrix`).

```python
non_zero = gt_rays_i[gt_rays_i > 0]
if non_zero.numel() >= 8:
    radius = torch.quantile(non_zero, 0.75)
elif non_zero.numel() > 0:
    radius = non_zero.max()
else:
    continue  # degenerate GT cell — skip assignment
containment_radius = radius * radius_scale
```

`radius_scale = 1.5` is the starting value. Monitor mean positive assignments per GT cell for the first 100 batches. Target: 1–4. Below 1 → scale too small. Above 10 → scale too large.

**Dense TIL tuning:** For tightly packed TIL fields, anchors from one cell can "steal" assignments for adjacent cells if the containment radius is too generous. If the model consistently misses small cells in dense clusters, reduce `radius_scale` to 1.2 or 1.3. This is a hyperparameter to tune after initial training, not before.

### 11.4 Pairwise IoU Helper

`polar_iou_pairwise_flat_torch(d_pred, d_gt)` lives in `raycasted/data/etl/ops/iou.py` and is shared with the loss function via `polar_iou_torch`. It accepts `[N_cand, N_gt, 32]` and returns `[N_cand, N_gt]`. It must not be reimplemented in `tal.py`. It accepts `[N_cand, N_gt, 32]` and returns `[N_cand, N_gt]`. It must not be reimplemented in `tal.py`.

The `_flat` suffix is meaningful: it signals that the batch dimension has already been collapsed by the caller's per-batch loop. The function never sees the `B` dimension. See §7.1 for the full naming contract.

---

## 12. Module D — ETL Pipeline

### 12.1 Two-Stage Design

| Stage | Files | Coord space | When |
|-------|-------|------------|------|
| **Ingestion** | Ingestors + `IngestionOrchestrator` | Pixel | Once per dataset |
| **Transform** | `SpatialChunker` + `NormalizerAndPadder` + `TransformOrchestrator` | Pixel | Once before training |

Both stages write `.npz` files. Ingestion produces full-ROI `.npz`. Transform produces tiled `.npz` with `content_h/w`.

### 12.2 IngestionOrchestrator

**File:** `raycasted/data/etl/ingestors/ingestion_orchestrator.py`

**Ingestor dispatch map:**

| `ingestion_method` | Ingestor class | Dataset |
|--------------------|---------------|---------|
| 1 | `ParquetIngestor` | MoNuSAC |
| 3 | `MatInstIngestor` | CoNSeP (currently commented out in YAML — must still be registered; see GAP-01) |
| 4 | `GeoJSONIngestor` | PUMA |
| 5 | `CSVPolygonIngestor` | PanopTILs |

Output layout: `<output_dir>/<dataset_name>/<split>/<roi_id>.npz`

**Generator vs return handling:** `ParquetIngestor.process_item()` is a generator (yields one tuple per ROI inside the Parquet file). `GeoJSONIngestor` and `CSVPolygonIngestor` use `return`. The orchestrator must detect both uniformly and iterate them the same way.

**`split` column:** The split label is read from `row['split']` — this is the column name produced by `BaseDataIngestor._build_registry()` (verified in existing code).

### 12.3 Raycast Annotation Extraction — Per-Ingestor Strategy

All three ingestors must implement `_extract_raycast_annotations()`. The shared geometry logic lives in `ops/convert.py` → `polygon_to_raycast()`.

**GeoJSON (PUMA):**
- Source: polygon vertex coordinates in `features[].geometry.coordinates[0]`
- Centroid: Shapely area-weighted `.centroid` (guaranteed inside polygon; vertex average is not)
- Handles both `Polygon` and `MultiPolygon` geometry types

**CSV Polygons (PanopTILs):**
- Source: comma-separated `coords_x` / `coords_y` strings, one row per cell
- Centroid: same Shapely centroid approach

**Parquet (MoNuSAC):**
- Source: per-cell binary PNG masks (not vertex coordinates)
- Contour extraction: `cv2.findContours(..., CHAIN_APPROX_NONE)` — use NONE not SIMPLE. SIMPLE suppresses collinear edge pixels, producing sparser Shapely geometries and causing ray misses on small circular cells (see NOTE-01)
- Centroid: OpenCV moments (`m10/m00`, `m01/m00`) — robust for non-convex masks
- Use the largest contour only; ignore spurious noise contours

### 12.4 `polygon_to_raycast` (ops/convert.py)

Core implementation requirements:

- Call `shapely.prepare(poly)` once before the 32-ray loop — builds spatial index, ~5× faster per-ray query.
- Use dynamic `R_far` tied to the polygon's bounding box:
  ```
  R_far = sqrt(bbox_w² + bbox_h²) × 1.1
  ```
  This is the bounding-box diagonal plus a 10% margin. It is the tightest value guaranteed to reach any point on the polygon boundary regardless of shape. Do not use a hardcoded constant — it fails for small cells due to floating-point precision, and wastes intersection computation for large cells.
- On non-convex boundaries a ray may cross the polygon boundary multiple times; take the **nearest** intersection to the centroid.
- Return `d_i = 0.0` for rays that do not intersect; downstream `filter_and_clip_annotations` handles survival filtering.

**Centroid placement and the `representative_point()` fallback:**

For severely concave or C-shaped cells, Shapely's area-weighted `.centroid` can fall outside the physical polygon boundary. A centroid outside the polygon means all 32 rays fail to intersect the boundary from inside, producing an all-zero annotation that `filter_and_clip_annotations` then discards.

Before casting any rays, check `poly.contains(Point(cx, cy))`. If the centroid is outside the polygon, fall back to `poly.representative_point()` — a Shapely method guaranteed to return a point strictly inside the geometry.

**Critical: when the fallback fires, `cx` and `cy` must be updated to the fallback point's coordinates before rays are cast AND before the annotation is written.** If the original centroid is kept in columns 1–2 while rays were cast from the representative point, the decoded polygon at inference will be offset from the stored centroid by the displacement between the two points. All 32 vertices are computed as `(cx + d_i × cos θ_i, cy + d_i × sin θ_i)` — if `cx/cy` doesn't match the ray-casting origin, the polygon shape will be geometrically incorrect in proportion to that displacement.

```python
def polygon_to_raycast(
    poly:              shapely.geometry.Polygon,
    class_id:          int,
    n_rays:            int = 32,
    fallback_counter:  collections.Counter | None = None,
) -> np.ndarray | None:
    """
    Converts a Shapely polygon to a raycast annotation row.

    Returns:
        np.ndarray of shape (35,) in unified format:
            [class_id, cx, cy, d_1, ..., d_32]  — pixel space
        None if the polygon has zero area or is unrecoverable.

    Centroid policy:
        1. Try area-weighted Shapely centroid.
        2. If centroid is outside polygon: fall back to representative_point().
        3. Update cx, cy to whichever point is used for ray casting.
           Never cast rays from one point and store a different centroid.

    Args:
        poly: Shapely Polygon to convert.
        class_id: Integer class label for the annotation.
        n_rays: Number of radial rays (default 32).
        fallback_counter: Optional `collections.Counter` for diagnostic logging.
            If provided, increments key 'representative_point_fallback' when the
            fallback is triggered. Pass `None` (default) for pure/test usage.
    """
    from shapely.geometry import Point, LineString
    import shapely

    if poly.area == 0:
        return None

    poly = poly.buffer(0)   # heal self-intersections
    if not poly.is_valid:
        return None

    # --- Centroid selection ---
    centroid = poly.centroid
    cx, cy   = centroid.x, centroid.y

    if not poly.contains(Point(cx, cy)):
        rep  = poly.representative_point()
        cx   = rep.x    # ← UPDATE cx/cy before ray casting
        cy   = rep.y    # ← this is the critical fix vs v4.6
        if fallback_counter is not None:
            fallback_counter['representative_point_fallback'] += 1

    # --- R_far: bounding-box diagonal + 10% ---
    minx, miny, maxx, maxy = poly.bounds
    bbox_w = maxx - minx
    bbox_h = maxy - miny
    R_far  = math.sqrt(bbox_w**2 + bbox_h**2) * 1.1

    # --- Ray casting ---
    shapely.prepare(poly)   # build spatial index once; ~5× faster per-ray query
    rays = np.zeros(n_rays, dtype=np.float32)

    angles = RAY_ANGLES   # from raycasted.data.utils.constants

    for i, theta in enumerate(angles):
        dx      = math.cos(theta) * R_far
        dy      = math.sin(theta) * R_far
        ray_line = LineString([(cx, cy), (cx + dx, cy + dy)])

        intersection = poly.boundary.intersection(ray_line)
        if intersection.is_empty:
            rays[i] = 0.0
            continue

        # For non-convex polygons the intersection may be a MultiPoint.
        # Take the nearest point to the centroid.
        if intersection.geom_type == 'Point':
            ix, iy = intersection.x, intersection.y
        elif intersection.geom_type == 'LineString':
            # Ray is tangent to the boundary — take the nearest endpoint to centroid.
            # LineString.coords returns (x, y) tuples, not Point objects.
            nearest = min(
                intersection.coords,
                key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2
            )
            ix, iy = nearest[0], nearest[1]
        else:
            # MultiPoint or GeometryCollection — iterate and find nearest
            pts = (
                list(intersection.geoms)
                if hasattr(intersection, 'geoms')
                else [intersection]
            )
            nearest = min(pts, key=lambda p: (p.x - cx)**2 + (p.y - cy)**2)
            ix, iy  = nearest.x, nearest.y

        rays[i] = math.sqrt((ix - cx)**2 + (iy - cy)**2)

    # Build unified-format annotation row: [class_id, cx, cy, d_1..d_32]
    row         = np.zeros(n_rays + 3, dtype=np.float32)
    row[0]      = float(class_id)
    row[1]      = cx    # updated centroid (may be representative_point if fallback fired)
    row[2]      = cy
    row[3:]     = rays

    return row
```

Log a diagnostic counter in each ingestor for how often the `representative_point()` fallback is triggered. The fraction of cells requiring the fallback should be below 5% per dataset. Above 5% indicates either unusual cell morphology or a systematic issue with source annotation quality (e.g., contours traced around cellular debris). See NOTE-04 for zero-ray monitoring guidance.

### 12.5 SpatialChunker Patch

Add `raycast` routing case to `_slice_annotations()`. Delegate entirely to `filter_and_clip_annotations()` from `ops/filter.py`. Do not reimplement clipping logic.

**Current code** (`spatialChunker.py`) only handles `bbox` and `instance_mask`. Add:
```python
elif self.annotation_type == 'raycast':
    return filter_and_clip_annotations(
        annotations, x_start, y_start, chunk_w, chunk_h, min_rays_after_clip=0.5
    )
```

Policy: `min_rays_after_clip=0.5` (strict — ETL decisions are permanent).

### 12.5.1 Filtering Pipeline Order (Critical)

The `filter_and_clip_annotations()` function must apply operations in the following **strict order** to ensure correct behavior:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: annotations [N, 35] in pixel space                                  │
│        [class_id, cx, cy, d_1...d_32]                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Filter by centroid position                                        │
│  - Drop cells where centroid (indices 1–2) is outside the crop region       │
│    [x_start, x_start+chunk_w) × [y_start, y_start+chunk_h)                  │
│  - Rationale: Cannot recover a cell whose center is outside the tile        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Translate coordinates to crop-relative space                       │
│  - cx' = cx (index 1) - x_start                                             │
│  - cy' = cy (index 2) - y_start                                             │
│  - Rays (indices 3–34) remain unchanged (distances from centroid)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Clip rays to crop boundary                                         │
│  - For each ray direction, compute max allowable distance:                  │
│    d_max = boundary_distance(cx', cy', angle_i, chunk_w, chunk_h)           │
│  - Set d_i = min(d_i, d_max) for all 32 rays                               │
│  - Note: Rays that were zero from polygon_to_raycast remain zero            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Filter by ray survival rate                                        │
│  - Count surviving rays: n_surviving = count(d_i > 0)                      │
│  - Survival rate = n_surviving / 32                                        │
│  - Drop cell if survival rate < min_rays_after_clip                        │
│  - Rationale: Partially-clipped cells with < 50% surviving rays             │
│    have insufficient shape information for training                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: filtered annotations [M, 35] in crop-relative pixel space          │
│         where M ≤ N and all cells meet survival threshold                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation notes:**
- Zero rays from `polygon_to_raycast` (geometry failures) are handled in STEP 4, not STEP 3
- A cell with many zero rays pre-clip may still pass if clipping introduces no additional zeros
- The survival threshold differs: ETL uses `0.5` (strict), DataLoader uses `0.3` (permissive)

### 12.6 NormalizerAndPadder Patch

**Current return signature** (`normalizer.py`, line 28):
```python
def process_roi(self, image: np.ndarray, annotations: Any) -> tuple[np.ndarray, Any]:
    ...
    return image, annotations  # Only 2 values — MUST BE UPDATED
```

**Required change:**
```python
def process_roi(self, image: np.ndarray, annotations: Any) -> tuple[np.ndarray, Any, int, int]:
    """Executes the Stage 3 transformation sequentially in memory.
    Returns: (transformed_image, untouched_annotations, content_h, content_w)
    """
    h, w = image.shape[:2]  # Store original dimensions
    
    # 1. Normalize
    if self.use_normalization and self.method == 'macenko':
        image = self._apply_macenko(image)

    # 2. Pad (if smaller than target size)
    if h < self.target_size or w < self.target_size:
        image = self._pad_bottom_right(image)
        # Annotations require NO changes because (0,0) remains top-left!

    return image, annotations, h, w  # Return original dimensions before padding
```

### 12.7 TransformOrchestrator Patch

**Current code** (`transform_orchestrator.py`, lines 42-48):
```python
# Run the memory-only Stage 3 transformer
final_img, final_annotations = self.normalizer.process_roi(img, annotations)

# Save to the Final PyTorch-Ready Directory
save_path = self.final_output_dir / npz_path.name
np.savez_compressed(save_path, image=final_img, bboxes=final_annotations, tissue=tissue)
# Missing: content_h, content_w; uses 'bboxes' instead of 'annotations'
```

**Required change:**
```python
# Load with backward compatibility
annotations = data.get('annotations', data.get('bboxes'))

# Run the memory-only Stage 3 transformer
final_img, final_annotations, content_h, content_w = self.normalizer.process_roi(img, annotations)

# Save to the Final PyTorch-Ready Directory
save_path = self.final_output_dir / npz_path.name
np.savez_compressed(
    save_path,
    image=final_img,
    annotations=final_annotations,  # Use 'annotations' key consistently
    tissue=tissue,
    content_h=np.int32(content_h),
    content_w=np.int32(content_w)
)
```

---

## 13. Module E — DataLoader

**File:** `raycasted/data/loader/raycast_dataset.py`

### 13.1 RayCastTileDataset

Reads tiled `.npz` files from `TransformOrchestrator` output. Applies online augmentation. Emits YOLO-format normalised tensors.

**Key design requirements:**

- `_random_crop()`: constrain crop origin to `[0, content_w − crop_size] × [0, content_h − crop_size]` using stored `content_h/w`
- `_normalise()`: divide all spatial quantities (cx, cy, rays) by `crop_size`. This is the **only** place normalisation happens.
- `_validate_batch()`: assert `labels[:, RAY_START_IDX:RAY_END_IDX].max() <= 1.0` (rays normalised) and `labels[:, CX_IDX:CY_IDX+1]` in `[0, 1]` (centroids normalised). See BUG-08.

### 13.2 Augmentation Contract

All augmentation functions live in `ops/augment.py`. They operate on **pixel-space** annotations and return **pixel-space** annotations. The DataLoader calls them before normalisation.

| Augmentation | Function | Permutation indices |
|--------------|----------|---------------------|
| Horizontal flip | `flip_horizontal(ann, canvas_w)` | `FLIP_H_IDX` |
| Vertical flip | `flip_vertical(ann, canvas_h)` | `FLIP_V_IDX` |
| 90° rotation | `rotate_90(ann, k, canvas_size)` | `ROT_IDX` |

**Signature contract (GAP-04):** Permutation indices are internal constants imported from `constants.py`. They are **never** passed as function arguments.

### 13.3 Collate Function

```python
def collate_fn(batch: list[tuple[torch.Tensor, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a batch of (image, labels) tuples.

    Args:
        batch: List of (image_tensor, labels_array) tuples.
               image_tensor: [3, H, W] float32
               labels_array: [N, 35] float32, normalised

    Returns:
        images: [B, 3, H, W] float32
        targets: [sum_N, 36] float32 — collated with batch_idx column
    """
    images = torch.stack([item[0] for item in batch])

    target_list = []
    for batch_idx, (_, labels) in enumerate(batch):
        if labels.shape[0] == 0:
            continue
        batch_col = torch.full((labels.shape[0], 1), batch_idx, dtype=torch.float32)
        target_list.append(torch.cat([batch_col, torch.from_numpy(labels)], dim=1))

    if target_list:
        targets = torch.cat(target_list, dim=0)
    else:
        targets = torch.zeros((0, 36), dtype=torch.float32)

    return images, targets
```

---

## 14. Module F — Inference & Visualisation

**File:** `ultralytics/models/yolo/detect/predict.py`

### 14.1 RayCastPredictor

Decodes model output to pixel-space polygon vertices.

**Key steps:**
1. Run model forward pass → `[B, N_anchors, nc + 34]`
2. Extract xy offsets (Sigmoid) and rays (Softplus)
3. Decode xy to absolute image coordinates using anchor grid
4. Denormalise rays by multiplying by `crop_size` — **read from `model.training_args['crop_size']`** (stored in the checkpoint by the training loop; default 640px). Do not hardcode — a mismatch between training and inference crop size produces geometrically wrong polygons with no error raised.
5. Build polygon vertices: `(cx + d_i × cos θ_i, cy + d_i × sin θ_i)` for i in 0..31
6. Filter by confidence threshold (default 0.25)
7. Apply distance-based deduplication (no NMS — Hungarian matching is end-to-end)

**No NMS.** Hungarian matching is end-to-end trained. If rare duplicates appear on out-of-distribution data, apply distance-based deduplication: suppress predictions whose centres fall within `dedup_radius_px` of a higher-confidence prediction.

The radius must be scale-aware:

```python
dedup_radius_px = min(5, imgsz * 0.008)
```

At 640px input this gives 5px. At 40× magnification (0.25 µm/px) TILs are typically 20–60px in diameter, so a 5px dedup radius is well within the minimum cell radius and will not suppress legitimate adjacent cells. The previous formula `min(20, imgsz*0.03)` gave ~19px, which is approximately the radius of the smallest TILs — it would suppress legitimate neighbours in dense fields.

### 14.2 RayCastAnnotator

Visualises decoded polygons on images. Draws:
- Polygon outline (green for TILs)
- Centroid marker (red dot)
- Optional: ray visualization for debugging

---

## 15. Module G — Validation Metrics

**File:** `ultralytics/models/yolo/detect/val.py`

### 15.1 RayCastValidator

Computes polygon-level mAP using Shapely polygon IoU.

**Key metrics:**
- mAP@0.50: IoU threshold = 0.5
- mAP@0.75: IoU threshold = 0.75
- mAP@0.50:0.95: mean over `{0.50, 0.55, …, 0.95}`

**Baseline:** LSP-DETR published PanNuke F1@0.5 and mAP50/75.

**Critical metric nuance — centroid F1 vs mask IoU F1:**  
LSP-DETR evaluates F1 based on *centroid proximity* (Euclidean distance threshold µ), not Shapely polygon IoU. Computing your model's F1 based purely on Shapely polygon IoU ≥ 0.5 will produce scores that are not directly comparable to LSP-DETR's reported numbers and may appear artificially lower.

`RayCastValidator` must implement both metrics:
- `shapely_f1(iou_threshold)` — exact 2D polygon intersection, publishable standard
- `centroid_f1(distance_threshold_px)` — match predictions to GT by centroid Euclidean distance; use the same threshold µ as LSP-DETR for apples-to-apples comparison

Report both in all evaluation runs. Use `shapely_f1` as the primary metric for your own published results; use `centroid_f1` only for direct LSP-DETR comparisons.

**Critical evaluation nuance — polygon overlap and PanNuke:**  
PanNuke ground truth annotations are forced to be non-overlapping (watershed-derived instance masks). LSP-DETR applies a deterministic distance-transform watershed refinement during inference to match this constraint before evaluation.

RayCastED predicts biologically correct polygon boundaries that may legitimately overlap for touching cells. If you evaluate directly against PanNuke's non-overlapping GT without post-processing, overlapping predictions are penalised even when they are geometrically accurate.

Options:
1. **No post-processing (default):** Accept the metric penalty. Document the discrepancy explicitly in the evaluation section of your thesis.
2. **Optional watershed refinement:** Apply the same distance-transform refinement as LSP-DETR at evaluation time only — never during training. This makes the comparison fair but adds evaluation complexity.

Start with option 1. Switch to option 2 only if reviewers require strict LSP-DETR comparability.

### 15.2 Shapely Parallelism

`multiprocessing.Pool(processes=8)` for epoch-end IoU matrix computation. Target: < 5 minutes for 10k predictions. The IoU function passed to `pool.starmap` must be a module-level function (not a method) for pickle compatibility.

---

## 16. Module H — Deployment (NVIDIA Jetson)

### 16.1 Overview

Deploy the trained RayCastED model on NVIDIA Jetson (Orin / Xavier) for edge inference on WSI tiles. The pipeline is:

```
PyTorch checkpoint (.pt)
    → ONNX export (training machine)
    → ONNX validation (onnxruntime, training machine)
    → TensorRT engine build (on Jetson device — engines are GPU-architecture-specific)
    → FP16 inference (Jetson)
```

TensorRT engines are **not portable** across GPU architectures. Always build the engine on the target Jetson device (or use the same JetPack version in a cross-compilation container). Never ship a `.engine` file built on a desktop GPU.

### 16.2 ONNX Export

**File:** `raycasted/export/onnx_export.py` (NEW)

**Graph boundary:** The ONNX graph contains the full model forward pass up to and including the detection head's raw output. Post-processing (confidence thresholding, xy decoding, ray denormalisation, vertex construction, deduplication) is **not** in the ONNX graph — it runs in Python/C++ on the Jetson.

**Why exclude post-processing:** Post-processing uses `crop_size` from training config and pre-computed trigonometric constants (`RAY_COS`, `RAY_SIN`). Baking these into the ONNX graph creates a fragile coupling: changing `crop_size` or `N_RAYS` requires re-exporting. Keeping post-processing external allows runtime configuration.

**Export function:**

```python
def export_polygon_yolo_onnx(
    weights_path: str,
    output_path: str,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic_batch: bool = False,
) -> str:
    """Export RayCastED to ONNX.

    Args:
        weights_path: Path to .pt checkpoint.
        output_path: Path to write .onnx file.
        imgsz: Input image size (square).
        opset: ONNX opset version. 17 covers all ops used (GroupNorm, Softplus, SiLU).
        simplify: Run onnx-simplifier to fold constants and remove redundant nodes.
        dynamic_batch: If True, export with dynamic batch dimension (batch_size=1..N).

    Returns:
        Path to the exported .onnx file.
    """
```

**Key steps:**
1. Load checkpoint, set model to eval mode
2. Create dummy input: `torch.randn(1, 3, imgsz, imgsz)`
3. `torch.onnx.export()` with:
   - `opset_version=17`
   - `input_names=['images']`
   - `output_names=['output']`
   - `dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}` if `dynamic_batch=True`
4. Validate with `onnx.checker.check_model()`
5. If `simplify=True`, run `onnxsim.simplify()`
6. Save and return path

**ONNX output shape:** `[B, N_anchors, nc + 34]` — identical to the PyTorch head output. `N_anchors` depends on `imgsz` (e.g., 8400 for 640×640 with standard YOLO stride set {8, 16, 32}).

**Opset 17 rationale:** GroupNorm (opset 6), Softplus (opset 1), SiLU/Swish (opset 14), Sigmoid (opset 1) are all covered. Opset 17 is the safe minimum that avoids known TensorRT compatibility issues with older opsets.

### 16.3 ONNX Validation

Before deploying to Jetson, validate numerical equivalence on the training machine:

1. Run PyTorch model on N sample images → collect raw outputs
2. Run same images through `onnxruntime.InferenceSession` → collect ONNX outputs
3. Assert `np.allclose(pytorch_output, onnx_output, atol=1e-5, rtol=1e-4)`

Small FP32 divergence (<1e-5) is normal due to operator fusion. If divergence exceeds 1e-4 on any output element, the export is broken — do not deploy.

### 16.4 TensorRT Engine Build (On-Device)

**File:** `raycasted/deploy/build_engine.py` (NEW — runs on Jetson only)

Build the TensorRT engine from the ONNX file on the target Jetson:

```bash
# CLI option (simplest)
/usr/src/tensorrt/bin/trtexec \
    --onnx=polygon_yolo.onnx \
    --saveEngine=polygon_yolo.engine \
    --fp16 \
    --workspace=4096 \
    --verbose
```

Or programmatic build via `tensorrt.Builder` API for tighter control over:
- FP16 mode (default — Jetson Orin achieves ~2× throughput vs FP32 with <0.5% mAP drop on detection tasks)
- Workspace size (4 GB default; reduce to 2 GB on Jetson Nano)
- Dynamic batch size (min=1, opt=1, max=8 for WSI tile streaming)

**FP16 is the default precision.** INT8 quantization requires a calibration dataset and post-training quantization (PTQ) or quantization-aware training (QAT). Defer INT8 unless FP16 throughput is insufficient.

### 16.5 Jetson Inference Runtime

**File:** `raycasted/deploy/jetson_inference.py` (NEW — runs on Jetson only)

The inference runtime on Jetson:

1. **Load engine:** Deserialise `.engine` file via `tensorrt.Runtime`
2. **Allocate buffers:** Host and device memory for input/output tensors
3. **Preprocess:** Load tile → resize to `imgsz` → normalise [0,1] → CHW → contiguous float32/float16
4. **Execute:** `context.execute_v2(bindings)` or async `execute_async_v2`
5. **Post-process** (on host, NumPy):
   - Extract class logits, xy offsets, ray distances from `[1, N_anchors, nc+34]`
   - Apply Sigmoid to class logits and xy offsets
   - Apply Softplus to ray distances: `log(1 + exp(x))`
   - Decode xy to absolute pixel coordinates using anchor grid
   - Denormalise rays: `rays_px = rays_norm × crop_size`
   - Confidence threshold filter (default 0.25)
   - Distance-based deduplication (same logic as `RayCastPredictor`)
   - Build polygon vertices: `(cx + d_i × cos θ_i, cy + d_i × sin θ_i)`

**`crop_size` source:** Read from a sidecar metadata file exported alongside the ONNX model (e.g., `polygon_yolo_meta.json` containing `{"crop_size": 640, "nc": 5, "n_rays": 32, "ray_angles": [...]}`). Never hardcode.

**Anchor grid:** The anchor grid positions depend on `imgsz` and stride set. Pre-compute during engine load and cache — it does not change between inference calls at the same resolution.

### 16.6 Export Metadata Sidecar

The ONNX export step must also write a JSON sidecar file containing all parameters needed for post-processing:

```json
{
    "crop_size": 640,
    "imgsz": 640,
    "nc": 5,
    "n_rays": 32,
    "ray_angles": [0.0, 0.19635, ...],
    "ray_cos": [1.0, 0.98079, ...],
    "ray_sin": [0.0, 0.19509, ...],
    "strides": [8, 16, 32],
    "conf_threshold": 0.25,
    "dedup_radius_px": 5
}
```

This decouples the Jetson runtime from the training codebase — the Jetson deployment needs only the `.engine` file, the sidecar JSON, and `jetson_inference.py`.

### 16.7 Performance Targets

| Platform | Precision | Target throughput | Latency target |
|----------|-----------|-------------------|----------------|
| Jetson Orin (32GB) | FP16 | ≥ 30 tiles/sec at 640×640 | ≤ 33 ms/tile |
| Jetson Xavier NX | FP16 | ≥ 10 tiles/sec at 640×640 | ≤ 100 ms/tile |

These targets are estimates based on published YOLO benchmarks on Jetson. Actual throughput depends on model complexity (backbone size, number of heads). Profile with `trtexec --onnx=model.onnx --fp16 --iterations=100` to measure before optimising.

---

## 17. Module I — Training Integration

### 17.1 Overview

Wires all model components (head, loss, assigner, predictor, validator) into a trainable pipeline using the Ultralytics training infrastructure. Produces trained checkpoints for ONNX export (Phase 9) and Jetson deployment.

**File:** `raycasted/model/train.py` — `RayCastTrainer(DetectionTrainer)`

### 17.2 RayCastTrainer

Subclasses `DetectionTrainer` to integrate all custom components:

| Method | Override | Purpose |
|--------|----------|---------|
| `get_model()` | Registers `RayCastDetect` head via `register_raycast_head()` | Head registration |
| `get_validator()` | Returns `RayCastValidator` | Shapely polygon mAP |
| `build_dataset()` | Returns `RayCastTileDataset` | Custom data loading |
| Loss wiring | Uses `RayCastE2ELoss` via `loss_fn` parameter | 5-term polygon loss |
| `_setup_train()` | Disables mosaic/mixup, asserts GAP-06 constraints | Safety checks |

### 17.3 Model Construction

Two approaches for building the training model:

**Approach A — YAML config (preferred if `parse_model()` supports it):**

```yaml
# raycasted/configs/raycast_yolo.yaml
# Standard YOLOv11 backbone/neck + RayCastDetect head
head:
  - [[17, 20, 23], 1, RayCastDetect, [nc, 32]]  # nc classes, 32 rays
```

**Approach B — Programmatic head replacement (fallback):**

1. Load a pretrained YOLO model: `YOLO('yolo11n.pt')`
2. Replace the `Detect` head with `RayCastDetect`, copying backbone/neck weights
3. Reinitialise head with cell-size-aware biases (§9.6)

The frozenset limitation in `parse_model()` (noted in §9) means `RayCastDetect` may not be resolvable from YAML alone. `register_raycast_head()` handles this, but if it fails, use Approach B.

### 17.4 Loss Wiring

The Ultralytics `E2ELoss.__init__` accepts a `loss_fn` parameter. `RayCastE2ELoss` passes `RayCastDetectionLoss` which internally swaps `TaskAlignedAssigner` for `RayCastAssigner`. The full chain:

```
RayCastE2ELoss(E2ELoss)
    ├── one2many: RayCastDetectionLoss → RayCastAssigner
    └── one2one:  RayCastDetectionLoss → RayCastAssigner
```

No manual wiring is needed beyond passing `loss_fn=RayCastDetectionLoss` to `super().__init__()`. See §10.1 for the subclass pattern.

### 17.5 DataLoader Integration

Override `build_dataset()` to use `RayCastTileDataset`:

```python
def build_dataset(self, img_path, mode='train', batch=None):
    from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
    return RayCastTileDataset(
        npz_dir=img_path,
        crop_size=self.args.imgsz,
        augment=(mode == 'train'),
    )
```

### 17.6 Safety Assertions

Before the first training batch, assert:

```python
assert self.args.mosaic == 0.0, "Mosaic corrupts polygon targets (GAP-06)"
assert self.args.mixup == 0.0, "Mixup corrupts polygon targets (GAP-06)"
assert not self.model.model[-1].use_dfl, "DFL must be disabled for polygon head"
```

### 17.7 Checkpoint Metadata

Trained checkpoints must store metadata required by inference and export:

```python
model.training_args = {
    'crop_size': 640,
    'imgsz': 640,
    'nc': nc,
    'n_rays': 32,
    'strides': [8, 16, 32],
}
```

Read by:
- `RayCastPredictor` — ray denormalisation at inference
- `export_polygon_yolo_onnx()` — ONNX export metadata sidecar

### 17.8 MLflow Logging

Per-epoch metrics logged to MLflow:

| Metric | Source | Frequency |
|--------|--------|-----------|
| `L_cls`, `L_xy`, `L_L1`, `L_PolarIoU`, `L_smooth` | Loss | Per batch |
| `lambda_smooth`, `o2m_weight` | Loss.update() | Per epoch |
| `mAP50`, `mAP75`, `mAP50-95` | RayCastValidator | Per epoch |
| `centroid_f1` (multiple thresholds) | RayCastValidator | Per epoch |
| Learning rate | Optimizer | Per epoch |

### 17.9 CLI Usage

```bash
# Train from scratch
python -m raycasted.model.train \
    --config raycasted/configs/raycast_yolo.yaml \
    --data raycasted/configs/dataset.yaml \
    --epochs 100 --batch 16 --imgsz 640

# Resume training
python -m raycasted.model.train --resume runs/detect/train/weights/last.pt
```

---

## 18. Hyperparameter Reference

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| N_RAYS | 32 | `constants.py` | Fixed. 64 only if cell morphology requires. |
| Angular spacing | 11.25° | `constants.py` | Immutable |
| Ray activation | Softplus | `head.py` | Never ReLU — see §9.3 |
| Centroid activation | Sigmoid | `head.py` | Grid-cell-relative offset |
| Ray bias target | 15px | `head.py` | Cell-size-aware init at 0.25 MPP — see §9.6 |
| `self.no` | `nc + 34` | `loss.py` | Must override parent — see BUG-02 |
| λ_cls | 0.5 | `loss.py` | |
| λ_xy | 1.0 | `loss.py` | |
| λ_L1 | 1.0 | `loss.py` | |
| λ_PolarIoU | 2.0 | `loss.py` | Dominant regression signal |
| λ_smooth | 0.05 → 0 | `loss.py` | Annealed by `RayCastE2ELoss.update()` |
| TAL alpha | 0.5 | `tal.py` | |
| TAL beta | 6.0 | `tal.py` | |
| radius_scale | 1.5 | `tal.py` | Monitor: target 1–4 positive assignments/GT |
| Candidate radius | 75th-pct ray | `tal.py` | More robust than mean for non-circular cells |
| GroupNorm groups | 8 | `head.py` | Reduce to 4 if channels < 64 |
| Polar-IoU eps | 1e-7 | `ops/iou.py` | Denominator guard |
| Huber beta | 0.01 | `ops/loss.py` | L2→L1 transition |
| Crop size | 640 | `dataset.py` | Square — single scalar normalises all spatial quantities |
| `min_rays_after_clip` (ETL) | 0.5 | `spatialChunker.py` | Strict: permanent decision |
| `min_rays_after_clip` (Loader) | 0.3 | `dataset.py` | Permissive: retried next epoch |
| `min_center_margin_px` (Loader) | 3 | `dataset.py` | Prevents edge-hugging annotations |
| Conf threshold | 0.25 | `predict.py` | Inference default |
| Dedup radius | `min(5, imgsz*0.008)` | `predict.py` | Scale-aware, not hardcoded |
| o2m weight | 0.8 → 0.1 | `train.py` | Inherited from `E2ELoss.decay()` |
| o2o weight | 0.2 → 0.9 | `train.py` | Complement of o2m |
| λ_smooth anneal end epoch | 50 | `loss.py` | Via `RayCastE2ELoss.update()` |

---

## 19. Bug & Vulnerability Registry

Every entry here must be addressed during implementation. Entries are classified by the severity of the consequence if ignored.

---

### 🔴 CRITICAL — Silent corruption, guaranteed convergence failure, or OOM crash

---

#### BUG-01 — Bias introduced by `eps` in clipping denominators
**File:** `raycasted/data/etl/ops/filter.py` → `filter_and_clip_annotations()`

The `where` condition gates out near-zero cosine/sine values before the division, so the denominators are already safe. Adding `eps` to them biases the boundary distance calculation.

**Wrong pattern to avoid:**
```python
d_right = np.where(cos_a > eps, (crop_w - cx_c) / (cos_a + eps), np.inf)
```

**Correct pattern:**
```python
d_right = np.where(cos_a > eps, (crop_w - cx_c) / cos_a, np.inf)
```

Apply the same correction to all four directional distance calculations (`d_right`, `d_left`, `d_bottom`, `d_top`).

---

#### BUG-02 — `self.no` not overridden in `RayCastDetectionLoss`
**File:** `raycasted/model/loss.py`

`v8DetectionLoss.__init__()` sets `self.no = m.nc + m.reg_max * 4`. The polygon head has no `reg_max`. Any code that reads `self.no` to slice prediction tensors (e.g. during `parse_output`) will produce tensors of the wrong shape without raising an error.

**Required in `RayCastDetectionLoss.__init__()`:**
```python
self.no = m.nc + 34
self.use_dfl = False
```

---

#### BUG-03 — Target encoding mismatch: Sigmoid offsets vs absolute normalised coordinates
**File:** `raycasted/model/loss.py` (fix) — bug originates from `ultralytics/utils/loss.py` parent

The head's xy output is a grid-cell-relative Sigmoid offset in `[0, 1]`. GT centroids from the DataLoader are absolute normalised coordinates in `[0, 1]`. These have the same numerical range but are in different spaces. Computing `L_xy` directly between them will not converge.

**Fix:** Implement `RayCastDetectionLoss.decode_pred_xy()` to decode Sigmoid outputs to absolute normalised space before computing `L_xy`. Do not encode GT into relative space — that coupling is fragile.

---

#### BUG-04 — `preprocess()` not overridden — applies box-specific transforms to polar targets
**File:** `raycasted/model/loss.py` (fix) — bug originates from `ultralytics/utils/loss.py` parent

`v8DetectionLoss.preprocess()` calls `xywh2xyxy()` and scales coordinates by `imgsz[[1,0,1,0]]`. Both operations corrupt a 34-dim polar target tensor before it reaches the assigner.

`RayCastDetectionLoss.preprocess()` must be fully overridden to accept `(N, 36)` batch targets and return `[B, N_gt_max, 35]` with no coordinate scaling (DataLoader already normalises).

---

#### BUG-05 — VRAM explosion in pairwise Polar-IoU
**File:** `raycasted/model/tal.py` (fix) — bug originates from `ultralytics/utils/tal.py` parent

A naive `[B, N_anchors, N_gt, 32]` expansion for pairwise IoU occupies ~8.6 GB at standard training configuration (B=16, N_anchors=8400, N_gt=500). This causes an immediate OOM crash. For dense TIL fields, even the batch-loop fix (which reduces complexity to `[N_cand, N_gt, 32]` per image) can produce ~3.8 GB slices when `N_cand` approaches `N_anchors`.

**Two-phase fix:**
1. Apply the spatial candidate mask from `select_candidates_in_gts()` before any ray expansion. Compute IoU only for candidates inside a per-batch loop.
2. Within the per-image computation, chunk `N_cand` into slices of `MAX_FLAT_PAIRS // N_gt` anchors when `N_cand × N_gt > MAX_FLAT_PAIRS` (default `MAX_FLAT_PAIRS = 1_000_000`, giving ~128 MB per chunk).

Full implementation in §11.2. Memory budget must be profiled at three density levels (sparse/moderate/dense) before training begins.

---

#### BUG-06 — TAL assigner patched in only one of three required methods
**File:** `raycasted/model/tal.py` (fix) — bug originates from `ultralytics/utils/tal.py` parent

Overriding only `get_box_metrics()` leaves `select_candidates_in_gts()` performing box-containment checks on polar vectors (geometrically wrong). At least these two methods must be overridden together. `get_targets()` does NOT need overriding — the parent is dimension-agnostic (uses `gt_bboxes.shape[-1]` dynamically) and correctly handles 34-dim polygon targets. See §11.1.

---

#### BUG-07 — `RayCastAssigner` not wired into the E2ELoss framework
**File:** `raycasted/model/loss.py` (fix) — bug originates from `ultralytics/utils/loss.py` parent

`E2ELoss` takes `loss_fn` as a constructor argument and instantiates it twice. `TaskAlignedAssigner` is instantiated inside `v8DetectionLoss.__init__()`. Subclassing only `TaskAlignedAssigner` does not affect which assigner `E2ELoss` uses — it still instantiates `v8DetectionLoss`, which instantiates `TaskAlignedAssigner`.

The required subclass chain is:
```
RayCastDetectionLoss(v8DetectionLoss)   ← overrides __init__ to instantiate RayCastAssigner
RayCastE2ELoss(E2ELoss)                 ← passes loss_fn=RayCastDetectionLoss
```

**Explicit assigner-swap pattern:**
```python
class RayCastDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)         # sets self.assigner = TaskAlignedAssigner(...)
        self.assigner = RayCastAssigner(# immediately replace — before any forward pass
            num_classes=self.nc,
            topk=self.topk,
            alpha=self.alpha,
            beta=self.beta,
        )
        self.no      = model.model[-1].nc + 34   # must follow super().__init__
        self.use_dfl = False
```

The one-line replacement is the correct pattern. It is safe because all inherited attributes needed to construct `RayCastAssigner` (`nc`, `topk`, `alpha`, `beta`) have been set by `super().__init__()` before the replacement executes.

---

#### BUG-08 — `validate_batch` ray upper bound must be 1.0, not 1.5
**File:** `raycasted/data/loader/raycast_dataset.py`

`filter_and_clip_annotations()` clips all rays to the crop content boundary. After normalisation by `crop_size`, no ray can exceed `crop_size / crop_size = 1.0`. Using an upper bound of 1.5 in the assertion would silently pass batches with incorrectly normalised rays.

**Required assertion:**
```python
assert labels[:, 4:36].max() <= 1.0, "ray > 1.0: clipping or normalisation bug"
```

---

### 🟡 MEDIUM — Functional gaps or interface contracts that break at integration

---

#### GAP-01 — `MatInstIngestor` must be registered in the ingestion dispatch map
**File:** `raycasted/data/etl/ingestors/ingestion_orchestrator.py`

`mat_inst_ingestor.py` (`ingestion_method=3`) exists in the repo. CoNSeP is currently commented out in the YAML config, but the orchestrator must still register method 3. Without it, uncommenting CoNSeP produces a confusing `ValueError` rather than a clear "not implemented" message.

**Fix:** Include `3: MatInstIngestor` in the dispatch map. No `raycast` implementation is needed for `MatInstIngestor` at this stage.

---

#### GAP-02 — `split` column name ✅ RESOLVED
**File:** `raycasted/data/etl/ingestors/ingestion_orchestrator.py`

**Resolution:** The split column is named `'split'` in the Polars DataFrame produced by `BaseDataIngestor._build_registry()` (verified in existing code at line 58-60). Use `row['split']` to read the split label.

---

#### GAP-03 — `annotation_type` is global-only by design
**File:** `raycasted/data/etl/config.py`

`annotation_type` is declared in `GlobalSettings` and applies to all datasets in a single pipeline run. This is intentional. To ingest different datasets with different annotation types, run the pipeline separately with different YAML configs.

This is not a bug. It is documented here to prevent it from being re-raised as a missing feature.

---

#### GAP-04 — Flip/rotate function signatures: permutation indices are internal
**File:** `raycasted/data/etl/ops/augment.py`

Permutation index arrays (`FLIP_H_IDX`, `FLIP_V_IDX`, `ROT_INDICES`) must be imported from `constants.py` inside `ops/augment.py` and used internally. They must **not** be passed as arguments by callers.

**Correct signatures:**
```python
flip_horizontal(annotations: np.ndarray, canvas_w: int) -> np.ndarray
flip_vertical(annotations: np.ndarray, canvas_h: int) -> np.ndarray
rotate_90(annotations: np.ndarray, k: int, canvas_size: int) -> np.ndarray
```

**Do not implement** 3/4-argument versions that accept index arrays as parameters. Any caller that passes `FLIP_H_IDX` as an argument can accidentally pass a stale or wrong index array, violating the single-source-of-truth contract in §4.1.

---

#### GAP-05 — Smoothness annealing and E2ELoss o2m decay are active simultaneously
**File:** `raycasted/model/loss.py` (monitoring) — inherited from `ultralytics/utils/loss.py`

Both the inherited `o2m` decay (from `E2ELoss.update()`) and `lambda_smooth` annealing operate in the epoch 0–50 window. The absolute magnitude of `L_smooth` at λ=0.05 is small, so destructive interference is unlikely but not impossible.

**Action:** Log both `lambda_smooth` and `o2m_weight` to MLflow from epoch 1. If training instability appears in epochs 10–30, delay the start of smoothness annealing to epoch 20.

---

#### GAP-06 — Mosaic and mixup augmentations must be explicitly disabled
**File:** `ultralytics/models/yolo/detect/train.py` (and any Ultralytics augmentation config)

Ultralytics' built-in mosaic and mixup pipelines validate and transform annotations assuming `(x, y, w, h)` bounding box format. When `RayCastTileDataset` yields `[N, 36]` polar targets, any mosaic or mixup operation that runs before the custom DataLoader will corrupt the annotation tensor — rays will be treated as box coordinates, scaled, concatenated, and clipped in ways that destroy the polar format.

**Action:** Set `mosaic=0.0` and `mixup=0.0` in the training hyperparameters. This must be enforced at the config level, not assumed. Add an assertion in the training setup that confirms both are zero before the first batch.

---

### 🟢 LOW — Pre-emptive notes for implementation and debugging

---

#### NOTE-01 — `CHAIN_APPROX_NONE` vs `CHAIN_APPROX_SIMPLE` in Parquet ingestor
**File:** `raycasted/data/etl/ingestors/parquet_ingestor.py`

`CHAIN_APPROX_NONE` is required for correctness: it preserves all boundary pixels, producing a dense polygon for Shapely ray-casting. `CHAIN_APPROX_SIMPLE` suppresses collinear edge pixels, which causes ray misses on small circular cells (< 0.5px error but concentrated on the axes where contour points were suppressed).

If MoNuSAC ingestion is unexpectedly slow, switching to `CHAIN_APPROX_SIMPLE` is an acceptable performance trade-off.

---

#### NOTE-02 — `ProcessPoolExecutor` pickling risk in the ingestion orchestrator
**File:** `raycasted/data/etl/ingestors/ingestion_orchestrator.py`

When `num_workers > 1`, the orchestrator serialises `_process_dataset` as a bound method via `ProcessPoolExecutor`. If `BaseDataIngestor` holds non-picklable state (open file handles, database connections), multiprocessing fails with an opaque pickle error. Start with `num_workers=1` and increase only after confirming the ingestor instances are picklable.

---

#### NOTE-03 — Polar-IoU is a sector-area approximation, not exact 2D polygon IoU
**File:** `raycasted/data/etl/ops/iou.py`

`PolarIoU = Σ min(d,d')² / Σ max(d,d')²` approximates sector areas assuming uniform angular spacing. For near-circular TILs this is tight. For elongated or irregularly-shaped cells the approximation diverges from exact 2D polygon IoU.

`L_PolarIoU` (λ=2.0) is the dominant loss term. If per-step Polar-IoU consistently looks good but epoch-end Shapely mAP is unexpectedly low, this approximation is the likely explanation. Monitoring divergence between the two metrics from epoch 1 is required.

---

#### NOTE-04 — Zero rays from `polygon_to_raycast` indicate geometry failures
**File:** `raycasted/data/etl/ops/convert.py`

Any ray that does not intersect the polygon boundary returns `d_i = 0.0`. These cells are not immediately dropped — they pass through ingestion and are filtered during `filter_and_clip_annotations()` based on the ray survival fraction threshold.

Zero rays can indicate: near-point cells (too small for Shapely to intersect reliably), severely self-intersecting source polygons that `buffer(0)` could not heal, or a centroid that falls outside the polygon boundary for highly concave shapes. The last case is handled by the `representative_point()` fallback described in §12.4 — if the fallback is triggered, the cell is salvaged rather than discarded, reducing the zero-ray rate.

Add a diagnostic counter in each ingestor that logs the number of cells with more than 5 zero rays. This fraction should be below 1% per dataset. Higher rates indicate either a data quality problem in the source annotations or a centroid placement issue not caught by the `representative_point()` fallback.

---

## 20. Testing Checkpoints

### Phase 0 — Shared Utilities

- [x] `filter_and_clip_annotations`: centre outside crop → empty array
- [x] `filter_and_clip_annotations`: centre inside, rays clipped → all rays ≤ boundary distance with no eps bias (see BUG-01)
- [x] `polar_iou(x, x) == 1.0` for arbitrary positive `x`
- [x] `polar_iou(x, 2*x) == 0.25` (analytical: Σx² / Σ4x²)
- [x] `polar_iou_torch(x, x) == 1.0` — torch variant matches NumPy on same inputs
- [x] `polar_iou_pairwise_flat_torch`: shape `[N_cand, N_gt, 32]` → output `[N_cand, N_gt]`; diagonal entries equal 1.0 when `d_pred == d_gt`
- [x] `angular_smoothness_loss(ones_tensor) == 0.0`
- [x] `angular_smoothness_loss_torch(ones_tensor) == 0.0` — torch variant matches NumPy
- [x] `flip_horizontal(ann, w)` applied twice → original annotations
- [x] `rotate_90(ann, k=1) applied four times → original annotations (4 × 90° = 360°)`

### Phase 0.5 — Ingestor Verification (before any full ETL run)

- [x] `polygon_to_raycast` round-trip: polygon → rays → vertices; area overlap threshold is **dataset-dependent**:
    - MoNuSAC / PanNuke (roughly circular nuclei): ≥ 0.95
    - PanopTILs (irregular lymphocyte shapes):     ≥ 0.90
    - Use 0.90 as the universal fallback if per-dataset shapes are unknown at test time.

- [x] `representative_point()` fallback: given a known concave test polygon (manually construct a C-shaped polygon where `.centroid` falls outside), confirm that:
    - The output row's `cx/cy` (indices 1–2) match the representative point, not the original centroid.
    - The decoded polygon vertices reconstruct visually correctly when rendered on the polygon's source image.
    - The fallback counter increments by exactly 1 for this test case.

- [x] `R_far` validation: for a circular polygon of radius `r`, confirm that `R_far = sqrt((2r)² + (2r)²) × 1.1 = 2r√2 × 1.1 ≈ 3.11r` and that all 32 rays are non-zero (i.e., `R_far` reaches the boundary in every direction).

- [ ] Visual: overlay decoded GT rays on 10 H&E crops from each dataset — boundaries align with cell membranes  *(deferred to Phase 1 — requires real dataset)*
- [ ] No annotation has any `d_i` greater than the centroid's distance to the nearest image edge  *(deferred to Phase 1)*
- [ ] Fraction of cells with > 5 zero rays < 1% per dataset (see NOTE-04)  *(deferred to Phase 1)*

### Phase 1 — ETL Ingestion

- [x] All three ingestors produce `(N, 35)` float32 arrays for `annotation_type=raycast`
- [x] `IngestionOrchestrator` writes correct `<dataset>/<split>/` directory layout
- [x] Split subdirectories populated correctly (use `row['split']`)
- [x] `MatInstIngestor` registered in dispatch map without error (see GAP-01)  *(importable from module; dispatch map is Phase 1.5)*
- [ ] Diagnostic: each ingestor passes `fallback_counter` to `polygon_to_raycast()` and logs fallback rate (§12.4)  *(deferred — implement before first real-data run)*
- [ ] Diagnostic: each ingestor logs count of cells with > 5 zero rays; fraction < 1% per dataset (NOTE-04)  *(deferred — implement before first real-data run)*

### Phase 2 — ETL Transform

- [ ] `NormalizerAndPadder.process_roi()` returns 4 values (image, annotations, content_h, content_w)
- [ ] `content_h` and `content_w` are the original dimensions (before padding)
- [ ] `TransformOrchestrator` unpacks 4-tuple correctly
- [ ] Output `.npz` contains keys: `image`, `annotations`, `tissue`, `content_h`, `content_w`
- [ ] `content_h <= target_size` and `content_w <= target_size` for padded tiles
- [ ] Unpadded tiles have `content_h == image.shape[0]` and `content_w == image.shape[1]`
- [ ] SpatialChunker `raycast` branch: no centroid outside `[0, chunk_w) × [0, chunk_h)`

### Phase 3 — DataLoader

- [ ] `validate_batch()` passes for first 10 batches with no assertions triggered
- [ ] `labels[:, 4:36].max() ≤ 1.0` (see BUG-08)
- [ ] `labels[:, 2:4]` in `[0, 1]`
- [ ] Mean annotations per tile > 0 (crop origin constraint working)
- [ ] Augmented images are visually plausible H&E (not inverted, not grey)
- [ ] Annotation count is unchanged by augmentations (values change, count must not)
- [ ] Training config asserts `mosaic=0.0` and `mixup=0.0` before first batch (see GAP-06)

### Phase 4 — Model Head

- [x] Output shape: `[B, 34, N_anchors]` (note: channel-first layout matching ultralytics convention)
- [x] `ray_outputs.min() > 0` (Softplus active)
- [x] `xy_outputs` in `[0, 1]` (Sigmoid active)
- [x] `self.no == nc + 34` in `RayCastDetectionLoss` (see BUG-02)
- [x] DFL decoder absent from computational graph (`head.dfl` is `nn.Identity`)
- [x] `RayRefinementBlock` preserves shape, has residual connection, GroupNorm config correct
- [x] Inference path: decoded xy non-negative, rays positive, scores in [0, 1]
- [x] Training forward returns raw logits (no activations applied)
- [x] Decode round-trip: encode known polygons as logits → decode via `_inference` → verify xy/rays match across P3/P4/P5 scales
- [x] Bias init: ray biases decode to ~15px at all scales (cell-size-aware for 0.25 MPP)
- [x] Visual: activation sanity plots (sigmoid/softplus distributions, per-scale ray statistics)
- [x] Visual: decoded polygon spread (8400 polygons from random logits, spatial coverage, ray distributions)

### Phase 5–6 — Loss + Assigner

- [ ] `decode_pred_xy` output in `[0, 1]` for valid Sigmoid inputs and standard anchor grid
- [ ] All five loss sub-terms non-zero in first batch
- [ ] No NaN or Inf with `d_pred = 1e-6 * torch.ones(8, 32)`
- [ ] GPU memory during assigner call ≤ 4 GB (see BUG-05)
- [ ] Mean positive assignments per GT cell: 1–4 for first 100 batches
- [ ] `lambda_smooth = 0.05` at epoch 0, `lambda_smooth = 0.0` at epoch 50
- [ ] `L_PolarIoU` decreasing monotonically over first 10 epochs

### Phase 7–8 — Inference + Metrics

- [ ] Decoded vertices are in pixel space (not `[0, 1]`)
- [ ] Polygons align visually with cell boundaries on held-out test images
- [ ] No duplicate predictions for single cells
- [ ] Shapely IoU of two identical polygons = 1.0
- [ ] Shapely IoU of two non-overlapping polygons = 0.0
- [ ] Epoch-end validation completes in < 5 minutes for 10k predictions
- [ ] Divergence between per-step Polar-IoU and Shapely mAP < 0.15 at epoch 20
- [ ] mAP50 > 0.30 within 20 epochs on held-out validation set
- [ ] Both `shapely_f1` and `centroid_f1` are computed and logged each evaluation epoch (see §14.2)
- [ ] `centroid_f1` at LSP-DETR distance threshold µ matches expected range for direct comparison

### Phase 9 — ONNX Export + TensorRT Deployment

- [ ] ONNX export produces valid model (passes `onnx.checker.check_model`)
- [ ] ONNX output shape matches PyTorch: `[1, N_anchors, nc + 34]`
- [ ] PyTorch vs ONNX numerical agreement: `np.allclose(output, atol=1e-5, rtol=1e-4)`
- [ ] onnx-simplifier reduces node count without changing output
- [ ] Metadata sidecar JSON written alongside `.onnx` with `crop_size`, `nc`, `n_rays`, `ray_angles`, `strides`
- [ ] TensorRT engine builds on Jetson without errors (`trtexec --fp16`)
- [ ] TensorRT FP16 output matches PyTorch FP32 within FP16 tolerance (`atol=0.01`)
- [ ] Jetson inference end-to-end: image in → polygon vertices out (pixel space)
- [ ] Throughput target met: ≥ 30 tiles/sec on Jetson Orin (FP16, 640×640)

### Phase 10 — Training Integration

- [ ] Training runs for 2 epochs without errors (smoke test with synthetic data)
- [ ] All five loss sub-terms logged to MLflow and non-zero in first epoch
- [ ] `lambda_smooth` decreases from 0.05 across epochs (reaches 0.0 after epoch 50)
- [ ] Checkpoint saved with `training_args` containing `crop_size`, `nc`, `strides`
- [ ] Mosaic/mixup assertions fire if accidentally enabled (GAP-06)
- [ ] Validation uses `RayCastValidator` (Shapely polygon mAP, not box mAP)
- [ ] ONNX export works with trained checkpoint (unblocks Phase 9 export tests)
- [ ] Resumed training continues from correct epoch and loss state
```

---

