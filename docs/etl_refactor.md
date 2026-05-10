# Project Refactor Blueprint: Ingestion Layer Architecture

## 1. Objective and Architectural Paradigm

This document outlines a structural refactor of the ingestors directory. The goal is to
migrate from a monolithic/tightly-coupled ingestion design to a strict 3-Tiered
"Composition over Inheritance" Architecture.

### Core Directives

- **DO NOT** use Multiple Inheritance (Mixins) for file handling.
- **DO NOT** put file I/O logic (e.g., `cv2.imread`, `polars.read_parquet`) inside the
  base class or the dataset-specific classes.
- State (configs, mappings) must be strictly isolated to **Tier 1**.
- File operations and format-specific functions (I/O, decoding, contour extraction tools)
  must be strictly isolated to **Tier 2** (Stateless static toolkits separated by file
  type, organized under `file_handlers`).
- Schema parsing must be strictly isolated to **Tier 3**.
- GPU-batched ray casting is a **Tier 2** concern — it is a format-agnostic geometry
  operation that all ingestors share. See `docs/etl-gpu-raycast.md` for the full
  mathematical specification.

---

## 2. Directory Transformation

### Current State

```
.
├── ingestors
│   ├── __init__.py              (Exports: Parquet, GeoJSON, CSV — NOT MatInst)
│   ├── _base.py                 (Mixed state and I/O logic)
│   ├── csv_poly_ingestor.py     (Mixes I/O and schema parsing)
│   ├── geojson_ingestor.py      (Mixes I/O and schema parsing)
│   ├── ingestion_orchestrator.py (DISPATCH_MAP: {1→Parquet, 3→MatInst, 4→GeoJSON, 5→CSV})
│   ├── mat_inst_ingestor.py     (raycast extraction = NotImplementedError)
│   └── parquet_ingestor.py      (Mixes I/O and schema parsing)
```

**DISPATCH_MAP gap:** Method key `2` is unassigned. Method `3` (MatInst) is registered
in the orchestrator but **not exported** from `__init__.py`.

### Target State

```
.
├── ingestors
│   ├── _base.py                 (Tier 1: Stateful Orchestrator)
│   ├── file_handlers            (Tier 2: Format-Specific Toolkits)
│   │   ├── __init__.py
│   │   ├── image.py             (Uses cv2)
│   │   ├── geojson.py           (Uses orjson)
│   │   ├── parquet.py           (Uses polars, cv2)
│   │   ├── mat.py               (Uses scipy)
│   │   └── raycast_gpu.py       (Uses torch — GPU batched ray casting)
│   ├── dataset_parsers          (Tier 3: Format-specific parsers)
│   │   ├── parquet_parser.py    (Formerly parquet_ingestor.py — not PanNuke-specific)
│   │   ├── geojson_parser.py    (Formerly geojson_ingestor.py — not PUMA-specific)
│   │   ├── csv_poly_parser.py   (Formerly csv_poly_ingestor.py — not PanOptils-specific)
│   │   └── mat_inst_parser.py   (Formerly mat_inst_ingestor.py — not CoNSeP-specific)
│   ├── ingestion_orchestrator.py
│   └── __init__.py
```

### Tier 2 — `raycast_gpu.py` (NEW)

This file contains the GPU-batched analytical ray-edge intersection solver, replacing
the per-cell shapely `polygon_to_raycast` calls in all ingestors.

**Class:** `RayCastGPU` (stateless, all `@staticmethod`)

| Method | Description |
|--------|-------------|
| `batch_polygon_to_raycast(vertices, class_ids, n_rays)` | Main entry point. Accepts a list of `(K_i, 2)` numpy vertex arrays (variable-length polygons), batches them into padded GPU tensors, and returns `(N, 3 + n_rays)` float32 annotations. Uses lazy `import torch`. |
| `_pad_vertices(vertices_list)` | Pad variable-length polygons to `K_max` by repeating the last vertex. Returns padded array + validity mask. |
| `_compute_centroids(V_padded, mask)` | Area-weighted centroid via shoelace formula. Vectorized over all polygons. |
| `_point_in_polygon(centroids, V_padded, mask)` | Crossing-number test to detect exterior centroids. |
| `_fallback_centroid(centroids, V_padded, mask)` | For exterior centroids, snap to the closest polygon vertex. |
| `_solve_ray_intersections(centroids, V_padded, mask, D)` | Cramer's rule batch solve for all (cell, ray, edge) triples. Returns `(N, R, K_max)` distances. |

**Data flow through the handler:**

```
CPU (per cell, lightweight):            GPU (batched, heavy):
  mask → cv2.findContours                ┌─────────────────────────┐
  contour.squeeze() → (K_i, 2)    ──►   │ Pad → Centroid → PiP    │
  collect all cells in ROI              │ → Cramer solve → Filter  │
                                        │ → (N, 3+R) annotations  │
                                        └─────────────────────────┘
```

**Dependency:** `torch` (lazy import). No shapely dependency in this module.

**Fallback:** If `torch` is unavailable or no CUDA device is found, falls back to the
existing shapely-based `polygon_to_raycast` from `ops/convert.py`. This is transparent
to Tier 3 parsers — they call the same handler method regardless of backend.

---

## 3. Execution Phases

### Phase 1: Establish Tier 2 (The Format-Specific Toolkits)

**Action:** Create a new directory `ingestors/file_handlers/` and populate it with
distinct modules for each file format.

**Description:** These files will contain purely static classes that handle
byte-decoding, file reading, and format-specific math/extraction. They have no
`__init__` methods and hold absolutely no state.

Implement the following files and classes:

#### `file_handlers/image.py`

- **Class:** `ImageHandler`
- **Contains:** `@staticmethod def load_rgb(path: str) -> np.ndarray` (uses cv2).
- **Reference:** Extract the `cv2.imread` logic previously found in
  `geojson_ingestor.py` or `mat_inst_ingestor.py`.

#### `file_handlers/geojson.py`

- **Class:** `GeoJSONHandler`
- **Contains:** `@staticmethod def load_json(path: str) -> dict` (uses orjson).
- **Reference:** Extract the JSON loading logic previously found in
  `geojson_ingestor.py`.

#### `file_handlers/parquet.py`

- **Class:** `ParquetHandler`
- **Contains:** `@staticmethod def decode_image_bytes(byte_string: bytes, is_mask: bool) -> np.ndarray`
  (uses `cv2.imdecode`).
- **Reference:** Extract the byte-decoding methods and structural Polars tricks
  previously found in `parquet_ingestor.py`.

#### `file_handlers/mat.py`

- **Class:** `MatHandler`
- **Contains:** `@staticmethod def load_mat(path: str) -> dict` (uses `scipy.io.loadmat`)
  and any static mathematical matrix operations.
- **Reference:** Extract the SciPy and `find_objects` matrix extraction logic
  previously found in `mat_inst_ingestor.py`.

#### `file_handlers/raycast_gpu.py`

- **Class:** `RayCastGPU`
- **Contains:** `@staticmethod def batch_polygon_to_raycast(vertices, class_ids, n_rays=32) -> np.ndarray`
  (uses `torch`, lazy import).
- **Reference:** New implementation based on the mathematical specification in
  `docs/etl-gpu-raycast.md`. Replaces per-cell `polygon_to_raycast` (shapely) calls
  in all Tier 3 parsers.
- **Fallback:** If CUDA is unavailable, falls back to serial `polygon_to_raycast`
  from `ops/convert.py`.

### Phase 2: Purify Tier 1 (The Stateful Base Class)

**Action:** Modify `ingestors/_base.py`.

**Description:** Ensure `BaseDataIngestor` is the single source of truth for pipeline
state.

Ensure these methods exist and are untouched:

- `__init__(self, config: dict)`: Holds `self.config`, `self.namespace_map`,
  `self.scale_factor`.
- `_build_registry(self)`: Uses polars for fast, vectorized file discovery and regex
  splits.
- `standardize_label(self, raw_label) -> int`: Maps dataset strings to global integers.
- `resolve_tissue(self, raw_tissue) -> int`: Maps dataset tissue to global integers.
- `standardize_mpp(self, image: np.ndarray, annotations: Any) -> tuple`: Applies
  `cv2.resize` and routes annotation scaling math based on the `annotation_type` config
  string.
- `@abstractmethod process_item(self, row: dict) -> tuple`: The required contract.

### Phase 3: Migrate to Tier 3 (Dataset-Specific Parsers)

**Action:** Move existing ingestor scripts into the new `ingestors/dataset_parsers/`
directory and rename them to reflect the specific dataset, not the file type.

**Description:** These classes must inherit **only** from `BaseDataIngestor`. They will
use Composition to call the toolkits from `ingestors/file_handlers/`.

#### Refactor Instructions

> **Naming convention:** Parsers are named by **format**, not by dataset. The same
> `ParquetParser` serves any parquet-based dataset (PanNuke or future datasets).
> Dataset-specific schema differences are handled inside `process_item()` via config-driven
> column mapping, not by subclassing per dataset.

**Rename `parquet_ingestor.py` → `dataset_parsers/parquet_parser.py`.**

- Class name: `ParquetParser(BaseDataIngestor)`.
- Use Polars to read the parquet file, extract bytes, and call
  `ParquetHandler.decode_image_bytes()`.
- For `annotation_type='raycast'`: collect contours via `cv2.findContours` per cell
  (CPU, lightweight), then call `RayCastGPU.batch_polygon_to_raycast()` for the entire
  ROI batch.
- The `ProcessPoolExecutor` spawn-based parallelism is preserved for ROI-level
  parallelism. Within each worker, the GPU ray cast is a single batch call.

**Rename `geojson_ingestor.py` → `dataset_parsers/geojson_parser.py`.**

- Class name: `GeoJSONParser(BaseDataIngestor)`.
- Inside `process_item`, call `ImageHandler.load_rgb(row['image_path'])` and
  `GeoJSONHandler.load_json(row['mask_path'])`.
- Schema navigation extracts polygon coordinates from GeoJSON features.
- For `annotation_type='raycast'`: collect all polygon vertex arrays from features, then
  call `RayCastGPU.batch_polygon_to_raycast(vertices, class_ids)` — **single GPU launch
  for the entire ROI** instead of per-cell shapely calls.

**Rename `csv_poly_ingestor.py` → `dataset_parsers/csv_poly_parser.py`.**

- Class name: `CSVPolyParser(BaseDataIngestor)`.
- Parse CSV coordinate columns into `(K, 2)` vertex arrays.
- For `annotation_type='raycast'`: batch all polygons and call
  `RayCastGPU.batch_polygon_to_raycast()`.

**Rename `mat_inst_ingestor.py` → `dataset_parsers/mat_inst_parser.py`.**

- Class name: `MatInstParser(BaseDataIngestor)`.
- Uses `scipy.io.loadmat` → `inst_map` instance matrix. Source format is **instance masks**
  (same raster→contour pattern as ParquetIngestor), not polygon coordinates.
- For `annotation_type='raycast'`: **currently a `NotImplementedError` stub.** Implementation
  follows the same pattern as ParquetIngestor: extract each instance ID's binary mask via
  `inst_map == instance_id`, run `cv2.findContours`, collect vertices, then batch via
  `RayCastGPU.batch_polygon_to_raycast()`.
- Also registered as `DISPATCH_MAP[3]` but **not exported** from `ingestors/__init__.py`
  — this must be fixed during migration.

#### Universal Mandate for All Tier 3 Parsers

Before yielding or returning the final tuple, every parser **MUST** call
`self.standardize_mpp(image_array, annotations)` to apply MPP scaling. This ensures
that all coordinate math is consistent regardless of the file format or the ray casting
backend (shapely vs GPU).

**GPU ray casting call pattern (unified across all parsers):**

```python
# Tier 3 parser — inside process_item, after extracting polygon vertices
vertices = []    # list of (K_i, 2) numpy arrays
class_ids = []   # list of int

for cell in cells:
    contour = extract_contour(cell)          # CPU — cv2.findContours or CSV parse
    if contour is not None and len(contour) >= 3:
        vertices.append(contour)
        class_ids.append(cell.class_id)

if vertices:
    annotations = RayCastGPU.batch_polygon_to_raycast(vertices, np.array(class_ids))
else:
    annotations = np.zeros((0, 3 + N_RAYS), dtype=np.float32)
```

This pattern is identical for all three file formats (parquet, geojson, csv) — the only
difference is how `extract_contour` produces the vertex array.

### Phase 4: String-Based Dispatch

Replace the integer-coded `DISPATCH_MAP` with a string-based registry that YAML configs
reference by name. This eliminates the opaque `ingestion_method: 1` codes.

**Config change — `DatasetConfig` in `utils/config.py`:**

```python
# Before (opaque integer codes):
ingestion_method: int    # 1=Parquet, 3=MatInst, 4=GeoJSON, 5=CSV

# After (self-documenting strings):
ingestor: str            # "parquet" | "mat_inst" | "geojson" | "csv_poly"
```

**Orchestrator change — `ingestion_orchestrator.py`:**

```python
# Before:
DISPATCH_MAP: dict[int, type] = {
    1: ParquetIngestor,
    3: MatInstanceIngestor,
    4: GeoJSONIngestor,
    5: CSVPolygonIngestor,
}

# After:
PARSER_REGISTRY: dict[str, type] = {
    'parquet':   ParquetParser,
    'mat_inst':  MatInstParser,
    'geojson':   GeoJSONParser,
    'csv_poly':  CSVPolyParser,
}
```

**YAML config migration (all `main/*.yaml`):**

```yaml
# Before:
datasets:
  PanNuke:
    ingestion_method: 1   # what does 1 mean?

# After:
datasets:
  PanNuke:
    ingestor: parquet     # self-documenting
```

**Done:** The legacy `ingestion_method` integer field has been fully removed from
`DatasetConfig`. All YAML configs now use the string `ingestor` field directly.

---

## 4. GPU Refactor Integration Notes

### Interaction with ProcessPoolExecutor

The current `parquet_ingestor.py` uses `ProcessPoolExecutor` with `spawn` start method
for ROI-level parallelism. The GPU refactor must account for this:

- **Option A (recommended):** Each worker process calls
  `RayCastGPU.batch_polygon_to_raycast()`. CUDA contexts are per-process — each spawn
  worker initializes its own. With ≤3 concurrent workers (PanNuke has 3 folds), this is
  manageable on a single GPU.
- **Option B:** Move GPU ray casting to the main process after all workers return
  contours. This avoids multi-process CUDA complexity but requires serializing contour
  data back from workers.

Option A is simpler and keeps the data flow local to each worker. The VRAM cost is
modest (~128MB per ROI batch, see `docs/etl-gpu-raycast.md` Section 8).

### Shapely Dependency

Shapely is **not removed** from the project. It is retained for:

1. `ops/convert.py` — `polygon_to_raycast` (backward compat, single-cell fallback).
2. `ops/convert.py` — `raycast_to_polygon` (inference path, used by `metrics.py`).
3. `tests/phase_0_5/test_round_trip.py` — validation tests.

The `RayCastGPU` handler uses shapely as a transparent fallback when CUDA is unavailable.
Tier 3 parsers never import shapely directly — they go through the handler.

### Cross-Reference

| Topic | Document |
|-------|----------|
| Mathematical specification (Cramer's rule, centroid, degenerate cases) | `docs/etl-gpu-raycast.md` Section 3 |
| Raster mask handling strategy | `docs/etl-gpu-raycast.md` Section 4 |
| Centroid fallback policy | `docs/etl-gpu-raycast.md` Section 5 |
| Performance estimates and VRAM budget | `docs/etl-gpu-raycast.md` Section 8 |
| Execution order for GPU refactor | `docs/etl-gpu-raycast.md` Section 7 |

---

## 5. Quality Assurance Checklist

Before completing the task, verify the following strict rules:

- [ ] **No Multiple Inheritance:** No Tier 3 class inherits from a file handler.
  (e.g., `class MyIngestor(BaseDataIngestor, ImageHandler):` is strictly forbidden).
- [ ] **Dependency Isolation:** `file_handlers/*.py` contains zero imports from
  `config.py` or internal state managers.
- [ ] **Correct Type Signature:** All Tier 3 parsers yield/return a 4-tuple matching
  exactly: `(roi_id: str, image_array: np.ndarray, annotations: Any, tissue_origin: int)`.
  Note that `tissue_origin` must be an `int`.
- [ ] **Empty Array Fallback:** If bounding boxes are completely empty in a patch, the
  array must be initialized safely as `np.empty((0, 5), dtype=np.int32)` to prevent
  downstream PyTorch slicing crashes.
- [x] **DISPATCH_MAP Migration:** The integer-based `DISPATCH_MAP` is replaced by
  a string-based `INGESTOR_REGISTRY`. All YAML configs use `ingestor: parquet`
  instead of `ingestion_method: 1`. The old integer keys and backward-compat
  validator have been removed.
- [x] **Config Validation:** `DatasetConfig.ingestor` type is `str`.
  Pydantic validates against the set of registered ingestor names.
- [ ] **MatInst Raycast Stub:** `mat_inst_parser.py`'s `_extract_raycast_annotations`
  is currently a `NotImplementedError`. When implementing, it uses the same
  raster→contour→GPU batch pattern as the parquet parser (instance mask pixels, not
  coordinate lists).
- [ ] **GPU Fallback:** `RayCastGPU.batch_polygon_to_raycast()` must gracefully degrade
  to the shapely backend when CUDA is unavailable, with a single warning log.
- [ ] **Round-Trip Accuracy:** GPU ray cast output must match shapely baseline within
  0.5px per ray (validated by `tests/phase_0_5/test_gpu_raycast.py`).
- [ ] **__init__.py Exports:** All Tier 3 parser classes referenced in `DISPATCH_MAP`
  must be exported from `ingestors/__init__.py` (currently `MatInstanceIngestor` is missing).

---

## 6. Loader: `RayCastTileDataset`

The loader sits at the end of the ETL pipeline, reading tiled `.npz` files produced by
`TransformOrchestrator` and emitting normalised tensors ready for training.

**Source:** `raycasted/data/etl/loader/raycast_dataset.py`

### 6.1 Architecture

```
.npz tiles on disk
    ↓  RayCastTileDataset.__getitem__()
image [H, W, 3] uint8  +  annotations [N, 3+n_rays] float32 (pixel space)
    ↓  1. Random crop (constrained to tissue area)
    ↓  2. Augmentation (geometric + photometric)
    ↓  3. Normalise (divide by crop_size → [0, 1])
    ↓  4. Validate (assert bounds)
image_tensor [3, H, W] float32  +  annotations [N, 3+n_rays] float32 (normalised)
    ↓  collate_fn()
images [B, 3, H, W]  +  targets [sum_M, 4+n_rays] (batch_idx prepended)
```

### 6.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pre-computed rays in `.npz` | Rays are computed once during ETL (not per-epoch). Faster epoch loop. |
| `crop_size`-constrained random crop | Crop origin clamped to `[0, content_{w,h} - crop_size]` so the crop stays on tissue, not white padding. |
| Single normalisation point | `_normalise()` is the **only** place where spatial values are divided by `crop_size`. Denormalisation at inference reads `crop_size` from `model.training_args`. |
| Ignore class filtered | `class_id=255` annotations are dropped in `__getitem__` and `_build_labels` to prevent index-out-of-bounds in the assigner. |
| Ultralytics label compat | `_build_labels()` creates a `labels` list with pseudo xyxy bboxes (cx/cy duplicated) so `plot_training_labels` doesn't crash. |

### 6.3 `__getitem__` Pipeline (5 stages)

1. **Load** — `np.load(path)` extracts `image`, `annotations`, `content_h`, `content_w`.
2. **Random crop** — `_random_crop()` selects a random `crop_size × crop_size` window
   within tissue bounds. Annotations outside the crop are filtered; rays crossing the
   crop boundary are clipped via `filter_and_clip_annotations()`. Tiles smaller than
   `crop_size` are padded with 255 (white).
3. **Augment** — `_augment()` applies in order: stain jitter (HSV), random scale,
   random translate, horizontal flip (50%), vertical flip (50%), rotation (0/90/180/270).
   Each geometric op transforms both image and annotations in lockstep. Short-circuits
   if annotations become empty mid-pipeline.
4. **Normalise** — `_normalise()` divides cx, cy, and all rays by `crop_size`. Class_id
   is left as integer.
5. **Validate** — `_validate_batch()` asserts rays ≤ 1.0 and centroids in [0, 1].

### 6.4 `collate_fn`

Prepends a `batch_idx` column to annotations from each sample, then concatenates across
the batch. Output shape: `targets [sum_M, 4+n_rays]` where `4 = batch_idx + class_id +
cx + cy`. Empty samples are skipped.

### 6.5 Comparison with LSP-DETR

RayCastED and LSP-DETR take fundamentally different approaches to data loading:

| Aspect | RayCastED | LSP-DETR |
|--------|-----------|----------|
| **Data source** | Pre-materialised `.npz` tiles on disk | HuggingFace `datasets` (streamed from Hub) |
| **Ray computation** | Pre-computed in ETL, stored in `.npz` | Computed online per-epoch via `stardist.star_distances()` |
| **Augmentation** | Custom ops (flip, rotate, scale, translate, stain jitter) | `albumentations` pipeline |
| **Collation** | Dense tensor with `batch_idx` column prepended | List-of-dicts (variable-length per sample) |
| **Framework** | Ultralytics `InfiniteDataLoader` | PyTorch Lightning + standard `DataLoader` |
| **Annotation format** | `[class_id, cx, cy, d_1, ..., d_n]` flat array | Dict with keys: `masks`, `labels`, `radial_distances`, `centroids` |
| **Class imbalance** | No explicit sampling strategy; relies on focal loss option in `RayCastDetectionLoss` | **`WeightedClassAndTissueSampler`** — inverse-frequency weighted sampling over tissue types AND cell classes |
| **Crop strategy** | Random crop constrained to tissue area | Full-image (no cropping — PanNuke images are 256×256) |

### 6.6 Class Imbalance — `WeightedClassSampler`

LSP-DETR's `WeightedClassAndTissueSampler` (`lsp_detr/data/samplers/weighted_class_and_tissue.py`)
addresses two axes of imbalance:

1. **Tissue balance** — Images of rare tissue types (e.g., PanNuke has 19 unevenly
   distributed tissue types) are upsampled using inverse-frequency weights:
   `w = N / (γ·count + (1-γ)·N)` where `γ=0.85`.
2. **Cell-class balance** — Images containing rare cell classes (e.g., few
   "dead" cells vs many "neoplastic" cells) are upsampled. Per-image weight is a
   weighted sum over the cell classes present, using the same inverse-frequency
   formula.

The two weight vectors are normalised and summed: `weight = tissue_w + class_w`.

**RayCastED now implements the same strategy** via `WeightedClassSampler`
(`raycasted/data/etl/loader/sampler.py`):

- `compute_tissue_weights(tissues, gamma)` — identical formula to LSP-DETR's
  `WeightedTissueSampler`. Reads the `tissue` field stored in each `.npz` tile by
  `TransformOrchestrator`.
- `compute_class_weights(tile_classes, num_classes, gamma)` — identical formula to
  LSP-DETR's `get_sampling_weights_cell`. Uses per-tile class histograms built from
  the `annotations[:, 0]` column (class IDs), excluding `class_id=255` (Ignore).
- `WeightedClassSampler` — a `WeightedRandomSampler` that sums both normalised weight
  vectors.

**Integration points:**

| Component | Change |
|-----------|--------|
| `RayCastTileDataset.__init__` | New `num_classes` param; lazy `_build_class_tissue_index()` on first sampler call |
| `RayCastTileDataset.get_sampler(gamma)` | Returns a `WeightedClassSampler` instance |
| `RayCastTrainer.build_dataset` | Passes `num_classes=head.nc` from the model head |
| `RayCastTrainer.get_dataloader` | Reads `weighted_sampling` and `sampler_gamma` from training config; swaps `shuffle=True` for `sampler=` when enabled |
| `main/pannuke.yaml` | New fields: `weighted_sampling: false`, `sampler_gamma: 0.85` |

**Usage — enable in YAML:**

```yaml
training:
  weighted_sampling: true
  sampler_gamma: 0.85    # 0 = uniform, 1 = full inverse-frequency
```

The sampler is **off by default** (`weighted_sampling: false`). When disabled, training
uses `shuffle=True` as before — no behavioural change. The `sampler` and `shuffle`
arguments are mutually exclusive in PyTorch's DataLoader, so the trainer selects one or
the other based on the config flag.

### 6.7 GPU Refactor Compatibility

The loader is **fully compatible** with the GPU refactor — no code changes needed. This is
by design: the GPU batched ray cast runs during ingestion (Stage 1), producing the same
`(N, 3+n_rays)` float32 annotation arrays that the shapely path produces. The loader reads
`.npz` files and is agnostic to how the annotations were computed.

**Data flow boundary:**

```
Ingestion (Stage 1)              Transform (Stage 2)           Loader (Stage 3)
┌──────────────────┐             ┌──────────────────┐         ┌───────────────┐
│ Raw masks/JSON   │             │ Chunk + pad      │         │ .npz tiles    │
│ → cv2 contours   │             │ → normalise      │         │ → augment     │
│ → RayCastGPU     │  .npz ROI   │ → filter/clip    │  .npz   │ → normalise   │
│   (or shapely)   │ ──────────► │                  │ ──────► │ → collate     │
│ → (N, 3+R) array │             │                  │         │ → [B,3,H,W]   │
└──────────────────┘             └──────────────────┘         └───────────────┘
     GPU lives here                  No GPU here                No GPU here
```

**Future consideration:** The `.npz` format currently stores no metadata about the ray
casting backend (shapely vs GPU). If provenance tracking becomes necessary (e.g.,
filtering cells where the GPU solver had ambiguous intersections), a `raycast_backend`
field could be added to the `.npz` without breaking the loader — it reads only `image`,
`annotations`, `content_h`, `content_w`.
