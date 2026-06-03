# ETL Refactor: GPU-Batched Ray Casting

Replace the shapely-based `polygon_to_raycast` in the **ingestion path** with a
batched analytical ray-edge intersection solver using PyTorch on GPU.

Branch: `etl-gpu-raycast` (from `exp`)

---

## 1. Problem Statement

During ingestion (`--stage ingest`), every cell polygon is converted to a raycast
annotation by `polygon_to_raycast()` (`raycasted/data/etl/ops/convert.py:17`).
This function:

1. Constructs a shapely `Polygon` from the contour vertices.
2. Computes the area-weighted centroid.
3. For each of 32 ray angles, builds a shapely `LineString` and calls
   `poly.boundary.intersection(ray_line)` — a GEOS C call.
4. Calls `nearest_points()` to find the closest intersection to the centroid.

For PanNuke (~200K cells × 32 rays = 6.4M GEOS intersection calls), this is the
dominant ingestion bottleneck. Each call incurs Python↔C FFI overhead, shapely
object allocation, and GEOS spatial index construction.

---

## 2. Scope

| Path | Change | Rationale |
|------|--------|-----------|
| `ops/convert_gpu.py` | **NEW** — batched GPU solver | Core algorithm |
| `ops/convert.py` | **UNCHANGED** — old shapely path stays | Backward compat, inference |
| `ops/__init__.py` | Add `batch_polygon_to_raycast` to exports | Public API |
| `ingestors/parquet_ingestor.py` | Route to GPU path | Main ingestor |
| `ingestors/geojson_ingestor.py` | Route to GPU path | Secondary ingestor |
| `ingestors/csv_poly_ingestor.py` | Route to GPU path | Secondary ingestor |
| `model/metrics.py` | **UNCHANGED** | Uses `raycast_to_polygon`, not `polygon_to_raycast` |
| `model/annotate.py` | **UNCHANGED** | Uses `decode_to_vertices`, not shapely |
| `tests/phase_0_5/test_round_trip.py` | **UNCHANGED** | Validates old path |
| `tests/phase_0_5/test_gpu_raycast.py` | **NEW** | Validates new path |

### Out of Scope

- Removing shapely from `pyproject.toml` (still needed by `raycast_to_polygon`,
  `metrics.py`, and `test_round_trip.py`).
- Modifying the transform stage — see `docs/etl-gpu-transform.md` for the
  companion GPU refactor of Macenko stain normalization.
- Modifying training or inference paths.

---

## 3. Mathematical Formulation

### 3.1 Definitions

| Symbol | Shape / Type | Description |
|--------|-------------|-------------|
| $N$ | `int` | Number of cells in the batch (per ROI) |
| $K_i$ | `int` | Number of vertices in polygon $i$ |
| $K_{\max}$ | `int` | $\max_i K_i$ — padded vertex count |
| $R$ | `int` | Number of rays ($= N_{\text{rays}}$, default 32) |
| $\theta_r$ | `float` | Angle of ray $r$: $\theta_r = r \cdot \frac{2\pi}{R}$ |
| $\mathbf{V}^{(i)}_j$ | $(2,)$ | Vertex $j$ of polygon $i$: $(x^{(i)}_j, y^{(i)}_j)$ |
| $\mathbf{P}^{(i)}$ | $(2,)$ | Centroid of polygon $i$: $(c^{(i)}_x, c^{(i)}_y)$ |
| $\mathbf{D}_r$ | $(2,)$ | Unit direction of ray $r$: $(\cos\theta_r,\ \sin\theta_r)$ |

### 3.2 Step 0 — Padding (CPU)

Each polygon $i$ has $K_i$ vertices of varying length. Pad to $K_{\max}$ by
repeating the last vertex:

$$\tilde{\mathbf{V}}^{(i)}_j = \begin{cases} \mathbf{V}^{(i)}_j & j < K_i \\ \mathbf{V}^{(i)}_{K_i - 1} & K_i \leq j < K_{\max} \end{cases}$$

A boolean mask tracks real vs padded vertices:

$$m^{(i)}_j = \begin{cases} 1 & j < K_i \\ 0 & K_i \leq j < K_{\max} \end{cases}$$

**Tensors moved to GPU:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\tilde{\mathbf{V}}$ | $(N,\ K_{\max},\ 2)$ | Padded vertices |
| $\mathbf{P}$ | $(N,\ 2)$ | Centroids |
| $\mathbf{m}$ | $(N,\ K_{\max})$ | Validity mask |
| $\mathbf{D}$ | $(R,\ 2)$ | Ray direction vectors |

### 3.3 Step 1 — Edge Vectors

For each cell $i$ and edge index $j$, the edge vector and offset from centroid:

$$\mathbf{E}^{(i)}_j = \tilde{\mathbf{V}}^{(i)}_{j+1} - \tilde{\mathbf{V}}^{(i)}_j$$

$$\mathbf{F}^{(i)}_j = \tilde{\mathbf{V}}^{(i)}_j - \mathbf{P}^{(i)}$$

where indices wrap: $\tilde{\mathbf{V}}^{(i)}_{K_{\max}} \equiv \tilde{\mathbf{V}}^{(i)}_0$.

**Implementation:** `torch.roll` on dim=1 with shift=-1, then subtract.

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{E}$ | $(N,\ K_{\max},\ 2)$ | Edge vectors |
| $\mathbf{F}$ | $(N,\ K_{\max},\ 2)$ | Centroid-to-vertex-start offsets |

### 3.4 Step 2 — Area-Weighted Centroid

Before ray casting, compute the centroid analytically (equivalent to shapely's
`poly.centroid` for simple polygons).

**Signed area** of polygon $i$ using the shoelace formula:

$$A^{(i)} = \frac{1}{2} \sum_{j=0}^{K_i - 1} \left( x^{(i)}_j \cdot y^{(i)}_{j+1} - x^{(i)}_{j+1} \cdot y^{(i)}_j \right)$$

**Centroid coordinates:**

$$c^{(i)}_x = \frac{1}{6\,A^{(i)}} \sum_{j=0}^{K_i - 1} \left( x^{(i)}_j + x^{(i)}_{j+1} \right) \left( x^{(i)}_j \cdot y^{(i)}_{j+1} - x^{(i)}_{j+1} \cdot y^{(i)}_j \right)$$

$$c^{(i)}_y = \frac{1}{6\,A^{(i)}} \sum_{j=0}^{K_i - 1} \left( y^{(i)}_j + y^{(i)}_{j+1} \right) \left( x^{(i)}_j \cdot y^{(i)}_{j+1} - x^{(i)}_{j+1} \cdot y^{(i)}_j \right)$$

Polygons with $|A^{(i)}| < \epsilon$ (zero area) are rejected.

### 3.5 Step 3 — Point-in-Polygon Test

Determine if centroid $\mathbf{P}^{(i)}$ lies inside polygon $i$ using the
**crossing number** (ray casting) algorithm.

For each edge $j$ of polygon $i$, cast a horizontal ray from $(c^{(i)}_x,\ c^{(i)}_y)$
to $(+\infty,\ c^{(i)}_y)$ and count edge crossings:

$$\text{cross}^{(i)}_j = \begin{cases} 1 & \text{if } \left(y^{(i)}_j > c^{(i)}_y\right) \neq \left(y^{(i)}_{j+1} > c^{(i)}_y\right) \\ & \text{and } c^{(i)}_x < x^{(i)}_j + \frac{c^{(i)}_y - y^{(i)}_j}{y^{(i)}_{j+1} - y^{(i)}_j} \left(x^{(i)}_{j+1} - x^{(i)}_j\right) \\ 0 & \text{otherwise} \end{cases}$$

$$\text{inside}^{(i)} = \left( \sum_{j=0}^{K_{\max}-1} \text{cross}^{(i)}_j \cdot m^{(i)}_j \right) \mod 2 = 1$$

**Fallback for exterior centroids:** If $\text{inside}^{(i)} = \text{False}$,
the centroid falls outside the polygon (concave shapes). The fallback uses
**binary search toward the mean of the 3 nearest vertices**:

1. Find the 3 real vertices $\{\mathbf{V}^{(i)}_{j_1}, \mathbf{V}^{(i)}_{j_2}, \mathbf{V}^{(i)}_{j_3}\}$ closest to $\mathbf{P}^{(i)}_{\text{area}}$ by Euclidean distance.
2. Compute the target: $\mathbf{T}^{(i)} = \frac{1}{3}(\mathbf{V}^{(i)}_{j_1} + \mathbf{V}^{(i)}_{j_2} + \mathbf{V}^{(i)}_{j_3})$
3. Probe fractions $\alpha \in \{0.5, 0.75, 0.875, 0.9375, 0.96875\}$ along the segment $[\mathbf{P}^{(i)}_{\text{area}}, \mathbf{T}^{(i)}]$:

$$\mathbf{Q}^{(i)}_\alpha = \mathbf{P}^{(i)}_{\text{area}} + \alpha \cdot (\mathbf{T}^{(i)} - \mathbf{P}^{(i)}_{\text{area}})$$

4. At each $\alpha$, test $\mathbf{Q}^{(i)}_\alpha \in \mathcal{P}$ using the crossing-number algorithm. Update the centroid only if the test succeeds.

After 5 iterations, the final precision is $1 - 0.96875 \approx 3.1\%$ of the original $\|\mathbf{T}^{(i)} - \mathbf{P}^{(i)}_{\text{area}}\|$ distance. The mean of 3 nearest vertices is chosen as the target because at least one vertex is typically on the interior-facing side of the polygon, making the mean likely interior. (Shapely uses `representative_point()` which is more precise but requires spatial index; the binary-search fallback is sufficient for histopathology cells where concavity is mild.)

### 3.6 Step 4 — Batched Ray-Edge Intersection (Core Solver)

This is the main computation. For every (cell, ray, edge) triple, solve the
2×2 linear system to find the ray-polygon boundary intersection.

**Parametric ray line** for cell $i$, ray $r$:

$$\mathbf{L}^{(i)}_r(t) = \mathbf{P}^{(i)} + t \cdot \mathbf{D}_r, \quad t \in [0,\ \infty)$$

**Parametric edge line** for cell $i$, edge $j$:

$$\mathbf{Q}^{(i)}_j(s) = \tilde{\mathbf{V}}^{(i)}_j + s \cdot \mathbf{E}^{(i)}_j, \quad s \in [0,\ 1]$$

**Set equal and solve for $(t, s)$:**

$$\mathbf{P}^{(i)} + t \cdot \mathbf{D}_r = \tilde{\mathbf{V}}^{(i)}_j + s \cdot \mathbf{E}^{(i)}_j$$

$$t \cdot \mathbf{D}_r - s \cdot \mathbf{E}^{(i)}_j = \mathbf{F}^{(i)}_j$$

This is a 2×2 system per (cell $i$, ray $r$, edge $j$):

$$\begin{bmatrix} D_{r,x} & -E^{(i)}_{j,x} \\ D_{r,y} & -E^{(i)}_{j,y} \end{bmatrix} \begin{bmatrix} t \\ s \end{bmatrix} = \begin{bmatrix} F^{(i)}_{j,x} \\ F^{(i)}_{j,y} \end{bmatrix}$$

**Solve via Cramer's rule** (optimal for 2×2 — no LU overhead):

$$\text{det}^{(i)}_{r,j} = D_{r,x} \cdot E^{(i)}_{j,y} - D_{r,y} \cdot E^{(i)}_{j,x}$$

$$t^{(i)}_{r,j} = \frac{F^{(i)}_{j,x} \cdot E^{(i)}_{j,y} - F^{(i)}_{j,y} \cdot E^{(i)}_{j,x}}{\text{det}^{(i)}_{r,j}}$$

$$s^{(i)}_{r,j} = \frac{F^{(i)}_{j,x} \cdot D_{r,y} - F^{(i)}_{j,y} \cdot D_{r,x}}{\text{det}^{(i)}_{r,j}}$$

Where the 2D cross product is defined as $a \times b = a_x b_y - a_y b_x$.

**Tensors after solving (all on GPU):**

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{t}$ | $(N,\ R,\ K_{\max})$ | Ray parameter for each (cell, ray, edge) |
| $\mathbf{s}$ | $(N,\ R,\ K_{\max})$ | Edge parameter for each (cell, ray, edge) |
| $\text{det}$ | $(N,\ R,\ K_{\max})$ | Determinant (for degenerate detection) |

### 3.7 Step 5 — Validity Filtering

A hit is valid when:

$$\text{valid}^{(i)}_{r,j} = \left( t^{(i)}_{r,j} > 0 \right) \wedge \left( -\epsilon \leq s^{(i)}_{r,j} \leq 1 + \epsilon \right) \wedge \left( m^{(i)}_j = 1 \right) \wedge \left( |\text{det}^{(i)}_{r,j}| > \epsilon_d \right)$$

where:
- $t > 0$: intersection is **in front of** the centroid (not behind).
- $0 \leq s \leq 1$: intersection falls **on the edge segment** (not past endpoints).
- $\epsilon = 10^{-6}$: boundary tolerance for vertex-hit numerical noise.
- $m^{(i)}_j = 1$: edge is real, not padded.
- $|\text{det}| > \epsilon_d$: ray is not parallel to the edge (avoids division-by-zero residue).

**Degenerate cases handled:**

| Case | Condition | Result |
|------|-----------|--------|
| Ray through vertex $\mathbf{V}_j$ | $s^{(i)}_{r,j-1} \approx 1.0$ and $s^{(i)}_{r,j} \approx 0.0$ | Both valid, same $t$ — `min` picks correctly |
| Ray tangent to edge | $\text{det}^{(i)}_{r,j} \approx 0$ | Filtered out by $|\text{det}| > \epsilon_d$ |
| Ray parallel, offset | $\text{det} = 0$, $\mathbf{D}_r \times \mathbf{F}^{(i)}_j \neq 0$ | $t = \pm\infty$, filtered by $t > 0$ and `isfinite` |
| Ray parallel, collinear | $\text{det} = 0$, $\mathbf{D}_r \times \mathbf{F}^{(i)}_j = 0$ | Edge lies along the ray — set $d_r = 0$ (conservative) |

### 3.8 Step 6 — Nearest Intersection Selection

For each cell $i$ and ray $r$, find the nearest valid hit:

$$\tilde{t}^{(i)}_{r,j} = \begin{cases} t^{(i)}_{r,j} & \text{if valid}^{(i)}_{r,j} = \text{True} \\ +\infty & \text{otherwise} \end{cases}$$

$$d^{(i)}_r = \min_{j} \tilde{t}^{(i)}_{r,j}$$

If no valid hits exist for ray $r$:

$$d^{(i)}_r = 0 \quad \text{(ray misses polygon entirely)}$$

**Implementation:** Set invalid $t$ to a large sentinel (`float('inf')`), then
`torch.min(dim=-1)` over the edge dimension.

### 3.9 Step 7 — Output Assembly

Package into the standard annotation format:

$$\mathbf{a}^{(i)} = \left[ \text{class\_id}^{(i)},\ c^{(i)}_x,\ c^{(i)}_y,\ d^{(i)}_0,\ d^{(i)}_1,\ \ldots,\ d^{(i)}_{R-1} \right]$$

Shape: $(N,\ 3 + R)$. Returned as `np.float32` on CPU.

---

## 4. Raster Mask Handling (C1 Strategy)

The ParquetIngestor receives per-cell binary masks (raster images), not polygon
coordinates. The conversion pipeline is:

### 4.1 Mask → Contour (CPU, per cell)

```python
mask_array = cv2.imdecode(mask_bytes, cv2.IMREAD_UNCHANGED)   # (H, W) binary
contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea)                  # (K, 1, 2)
V = contour.squeeze(axis=1)                                   # (K, 2) float
```

This step remains on CPU. `cv2.findContours` is already fast (~0.1ms/cell) and
has no GPU equivalent that would justify the data transfer overhead.

### 4.2 Contour → GPU Batch

After extracting contours for all cells in an ROI, batch them into a single
padded tensor and transfer to GPU for the analytical solver. This replaces the
current per-cell shapely ray cast with a single GPU kernel launch.

### 4.3 Special Case: Multi-Contour Cells

A cell mask may produce multiple contours (e.g., disconnected mask artifacts).
Current behavior: take the largest contour by area. The GPU path preserves this
by filtering `max(contours, key=cv2.contourArea)` before batching.

---

## 5. Centroid Policy

The current shapely implementation has a 3-step centroid policy:

1. Compute area-weighted centroid.
2. If centroid is outside polygon → fall back to `representative_point()`.
3. Update $c_x, c_y$ to whichever point is used for ray casting.

The GPU implementation replaces `representative_point()` (which requires GEOS
spatial index) with **binary search toward the mean of the 3 nearest vertices**
(Section 3.5). This is acceptable because:

- Histopathology cells are mildly concave at worst — the area-weighted centroid
  is almost always inside.
- The mean of 3 nearest vertices is very likely interior: at least one vertex
  is on the near side of the polygon, pulling the mean inward.
- Binary search (5 iterations) guarantees finding an interior point along the
  segment, with 3.1% final precision.
- Diagnostic counter preserved: increment `nearest_vertex_fallback` for monitoring.

---

## 6. Implementation Plan

### Phase 1: Core GPU Solver

**File:** `raycasted/data/etl/ops/convert_gpu.py`

```python
def batch_polygon_to_raycast(
    vertices: list[np.ndarray],   # [(K_0, 2), (K_1, 2), ...]
    class_ids: np.ndarray,        # (N,)
    n_rays: int = 32,
    device: str = 'cuda',
    fallback_counter: collections.Counter | None = None,
) -> np.ndarray:
    """Batch-analytical polygon-to-raycast conversion on GPU.

    Returns:
        annotations: (N, 3 + n_rays) float32, same format as polygon_to_raycast.
    """
```

Uses lazy `import torch` inside the function body (same convention as
`ops/iou.py` and `ops/loss.py`).

### Phase 2: Integrate into ParquetIngestor

**File:** `raycasted/data/etl/ingestors/parquet_ingestor.py`

Modify `_extract_raycast_annotations()`:

```
Current flow (per cell, serial):
    for each mask_row:
        mask → cv2.findContours → Polygon → polygon_to_raycast(shapely)

New flow (batched, GPU):
    contours = []
    for each mask_row:
        mask → cv2.findContours → contour.squeeze()   # CPU, lightweight
        contours.append(contour)
    annotations = batch_polygon_to_raycast(contours, class_ids)  # GPU, single launch
```

The Python loop still exists but is lightweight (no shapely, no ray casting).
The heavy work is a single GPU kernel launch.

### Phase 3: Integrate into GeoJSON/CSV Ingestors

**Files:** `geojson_ingestor.py`, `csv_poly_ingestor.py`

Same pattern: collect polygon vertices (already available as coordinate lists),
batch and call `batch_polygon_to_raycast`.

For GeoJSON, polygon vertices come directly from the JSON:
```python
coordinates = feature['geometry']['coordinates'][0]   # list of (x, y)
V = np.array(coordinates, dtype=np.float64)            # (K, 2)
```

For CSV, polygon vertices come from parsed coordinate columns:
```python
x_arr = np.array(x_str.split(','), dtype=np.float64)
y_arr = np.array(y_str.split(','), dtype=np.float64)
V = np.column_stack([x_arr, y_arr])                    # (K, 2)
```

### Phase 4: Update Public API

**File:** `raycasted/data/etl/ops/__init__.py`

Add `batch_polygon_to_raycast` to imports and `__all__`.

### Phase 5: Tests

**New file:** `tests/phase_0_5/test_gpu_raycast.py`

Test cases:

| Test | Description |
|------|-------------|
| `test_circle_exact` | Regular n-gon (64 vertices) → all rays should equal radius ±1% |
| `test_convex_round_trip` | Circle → GPU raycast → `raycast_to_polygon` → shapely IoU ≥ 0.95 |
| `test_concave_round_trip` | Star/L-shape → round-trip IoU ≥ 0.85 |
| `test_matches_shapely` | Same polygons → compare GPU vs shapely ray distances, max error < 0.5px |
| `test_empty_polygon` | Zero-area polygon → rejected (returns empty row) |
| `test_vertex_hit` | Construct polygon where a ray passes exactly through a vertex |
| `test_tangent_ray` | Construct polygon where a ray is tangent to an edge |
| `test_exterior_centroid` | Concave polygon with exterior centroid → fallback triggers |
| `test_batch_vs_serial` | Verify batched output matches serial `polygon_to_raycast` for 50 random polygons |

**Validation command:**

```bash
uv run python tests/phase_0_5/test_gpu_raycast.py
```

---

## 7. Execution Order

```
1. Create branch: etl-gpu-raycast
2. Implement ops/convert_gpu.py  (Phase 1)
3. Write tests/phase_0_5/test_gpu_raycast.py  (Phase 5)
4. Validate GPU solver against shapely baseline
5. Integrate into parquet_ingestor.py  (Phase 2)
6. Integrate into geojson_ingestor.py, csv_poly_ingestor.py  (Phase 3)
7. Update ops/__init__.py  (Phase 4)
8. Run full ETL pipeline test: uv run python -m raycasted.pipeline --config main/pannuke.yaml --output output/gpu_test --stage ingest
9. Compare output .npz files against shapely baseline
10. Run lint: uv run ruff check . && uv run ruff format .
```

---

## 8. Performance Expectations

| Metric | Current (shapely) | Expected (GPU batch) |
|--------|-------------------|---------------------|
| Per-cell cost | ~0.5ms (32 GEOS calls + Python overhead) | ~0.01ms (batched tensor ops) |
| PanNuke Fold3 (~65K cells) | ~30s | ~1s + data transfer |
| Memory | O(1) per cell (serial) | O(N × K_max × R) GPU VRAM |
| Accuracy | Exact (GEOS arbitrary precision) | < 0.5px error (float32) |

For a typical ROI with ~1000 cells and $K_{\max} \approx 500$ vertices:

$$\text{VRAM} = 1000 \times 500 \times 32 \times 2 \times 4\text{ bytes} \approx 128\text{ MB}$$

This fits comfortably on any GPU. For extremely large ROIs, the batch can be
split into sub-batches of ~5000 cells.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Float32 precision causes vertex-hit misses | $s$ tolerance $\epsilon = 10^{-6}$; tested with synthetic vertex-aligned rays |
| Concave polygon centroid outside | Binary search toward 3-nearest-vertex mean (Section 3.5); diagnostic counter |
| Self-intersecting polygon (from noisy mask) | Area filter ($|A| < \epsilon$ → reject); same as current `poly.buffer(0)` behavior |
| GPU not available (CI/VM) | Fallback to shapely path with a config flag or auto-detect |
| Large $K_{\max}$ wastes memory on simple cells | Percentile-based $K_{\max}$ (e.g., 95th percentile) with overflow batch |
