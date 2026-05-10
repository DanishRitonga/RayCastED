# ETL Refactor: GPU-Accelerated Transform Stage

Replace the NumPy-based Macenko stain estimation and normalization in the
**transform pipeline** with batched PyTorch operations on GPU.

Branch: `etl-gpu-raycast` (from `exp`) — companion to the ingestion GPU refactor
documented in `docs/etl-gpu-raycast.md`.

---

## 1. Problem Statement

The transform pipeline has three stages:

| Stage | Class | Current bottleneck |
|-------|-------|--------------------|
| 1. Chunking | `SpatialChunker` | Lightweight index math — **no GPU benefit** |
| 2. Profiling | `StainEstimator` | Dense linear algebra on ~1M pixels per tile, runs on **every** tile |
| 3. Normalization | `NormalizerAndPadder` | Same dense LA as Stage 2, plus trivial padding |

Stages 2 and 3 call Macenko's algorithm per tile, which involves:

1. RGB → Optical Density conversion (`-log10`) on all pixels (~1M for 1024×1024).
2. Covariance matrix + eigen decomposition (`np.cov` + `np.linalg.eigh`).
3. Matrix projections (`np.dot` on ~1M × 3 × 3).
4. Pseudo-inverse + concentration reconstruction (`np.linalg.pinv` + `np.dot`).

For PanNuke (~2000+ tiles), this produces **4000+ eigen decompositions** and
**4000+ pseudo-inverses**. Each operation is small (3×3 matrices) but the
per-tile overhead of Python dispatch and NumPy memory allocation dominates.

GPU batching eliminates the Python loop overhead and allows cuSOLVER/cuBLAS to
process many tiles in parallel.

---

## 2. Scope

| Path | Change | Rationale |
|------|--------|-----------|
| `transform/stainEstimator_gpu.py` | **NEW** — batched GPU Macenko | Core algorithm |
| `transform/stainEstimator.py` | **UNCHANGED** — old NumPy path stays | Backward compat, fallback |
| `transform/normalizer.py` | Add GPU normalization path | Routes to GPU estimator |
| `transform/transform_orchestrator.py` | Batch-aware orchestration | Batch tiles to GPU |
| `transform/spatialChunker.py` | **UNCHANGED** | No GPU benefit |
| `ops/filter.py` | **UNCHANGED** | Operates on raycast annotations, not images |
| `tests/phase_0_5/test_gpu_stain.py` | **NEW** | Validates GPU Macenko |

### Out of Scope

- Modifying `SpatialChunker` or `filter.py` — pure index/annotation ops.
- Modifying training or inference paths.
- Stain estimation methods beyond Macenko (Vahadane, Reinhard — currently `NotImplementedError`).

---

## 3. Mathematical Formulation

### 3.1 Macenko Algorithm (Per Image)

The existing NumPy implementation in `stainEstimator.py:20-80` follows these steps:

**Step 1 — RGB to Optical Density:**

$$\text{OD} = -\log_{10}\!\left(\frac{I + 1}{I_0}\right)$$

where $I$ is the $(H \cdot W, 3)$ reshaped image and $I_0 = 240$.

**Step 2 — Filter background:**

$$\hat{\text{OD}} = \text{OD}[\,\neg\,\text{any}(\beta > \text{OD},\ \text{axis}{=}1)\,]$$

where $\beta = 0.15$. If $|\hat{\text{OD}}| < 100$ pixels, return `(None, None)`.

**Step 3 — Eigendecomposition:**

$$C = \text{cov}(\hat{\text{OD}}),\quad \text{eigvals},\ \text{eigvecs} = \text{eigh}(C)$$

Select the two eigenvectors with largest eigenvalues: `eigvecs[:, [1, 2]]`.

**Step 4 — Angle projection:**

$$T = \hat{\text{OD}} \cdot E, \quad \phi = \arctan2(T[:, 1],\ T[:, 0])$$

**Step 5 — Robust extrema:**

$$\phi_{\min} = \text{percentile}(\phi,\ \alpha),\quad \phi_{\max} = \text{percentile}(\phi,\ 100 - \alpha)$$

$$\mathbf{v}_{\min} = E \cdot [\cos\phi_{\min},\ \sin\phi_{\min}]^T, \quad \mathbf{v}_{\max} = E \cdot [\cos\phi_{\max},\ \sin\phi_{\max}]^T$$

**Step 6 — H&E ordering:**

$$H_E = \begin{cases} [\mathbf{v}_{\min},\ \mathbf{v}_{\max}] & \text{if } v_{\min,x} > v_{\max,x} \\ [\mathbf{v}_{\max},\ \mathbf{v}_{\min}] & \text{otherwise} \end{cases}$$

Normalize: $\hat{H_E} = H_E \,/\, \|H_E\|$ (row-wise L2).

**Step 7 — Concentrations:**

$$C_{\text{hat}} = \hat{\text{OD}} \cdot \text{pinv}(\hat{H_E}),\quad \max_C = \text{percentile}(C_{\text{hat}},\ 99,\ \text{axis}{=}0)$$

Output: $(\hat{H_E},\ \max_C)$ — shape $(2, 3)$ and $(2,)$.

### 3.2 Normalization Transform (Per Image)

The existing `_apply_macenko` in `normalizer.py:51-80`:

1. Estimate source stain matrix and concentrations (same as above).
2. Compute concentrations: $C_{\text{src}} = \text{OD} \cdot \text{pinv}(H_E^{\text{src}})$.
3. Scale: $C_{\text{norm}} = C_{\text{src}} \cdot (\max_C^{\text{target}} \,/\, \max_C^{\text{src}})$.
4. Reconstruct: $\text{OD}_{\text{norm}} = C_{\text{norm}} \cdot H_E^{\text{target}}$.
5. Convert back: $I_{\text{norm}} = I_0 \cdot 10^{-\text{OD}_{\text{norm}}} - 1$, clip to `[0, 255]`.

### 3.3 Batched GPU Formulation

For a batch of $B$ images, each of size $H \times W \times 3$:

**Challenge:** Images may have different numbers of non-background pixels after
Step 2 filtering. This prevents simple stacking.

**Strategy:** Process images independently on GPU using `torch.vmap` or a padded
batch approach. Two options:

#### Option A: Sequential GPU (Recommended)

Transfer each image to GPU individually, run the full Macenko pipeline on GPU
tensors. The benefit comes from GPU-native `torch.linalg.eigh`, `torch.pinverse`,
and `torch.log10` being faster than NumPy for these small matrices — especially
when the data is already on GPU from ingestion.

```
for each tile:
    tensor = torch.from_numpy(image).cuda()       # ~0.1ms transfer
    OD, HE, maxC = macenko_gpu(tensor)             # ~0.5ms vs ~5ms NumPy
```

No batching complexity. ~10× speedup from eliminating NumPy Python dispatch
overhead alone. Best for the current tile-at-a-time orchestrator architecture.

#### Option B: Batched GPU (Future optimization)

Stack $B$ images into a $(B, H, W, 3)$ tensor. Process Steps 1-3 in parallel.
For Steps 4-7 (which have variable-length $\hat{\text{OD}}$), use a padded mask
approach:

1. Compute OD for all $B$ images: $(B, H \cdot W, 3)$.
2. Compute background mask: $(B, H \cdot W)$ boolean.
3. **Percentile trick:** Instead of filtering to variable-length $\hat{\text{OD}}$,
   compute the covariance using a masked mean:
   $$\mu = \frac{\sum_{\text{valid}} \hat{\text{OD}}}{|\text{valid}|}, \quad
     C = \frac{(\hat{\text{OD}} - \mu)^T \cdot M \cdot (\hat{\text{OD}} - \mu)}{|\text{valid}| - 1}$$
   where $M$ is a diagonal mask. This avoids variable-length indexing entirely.
4. `torch.linalg.eigh` on $(B, 3, 3)$ covariance matrices — batched.
5. Angle projection and percentile on masked pixels.
6. `torch.linalg.pinv` on $(B, 2, 3)$ stain matrices — batched.

This is more complex but allows processing 8-16 tiles simultaneously.

**Recommendation:** Start with Option A (sequential GPU). It gives the biggest
bang for the buck with minimal code change. Option B can be layered on later if
profiling shows the GPU is underutilized.

---

## 4. Implementation Plan

### Phase 1: GPU Stain Estimator

**File:** `raycasted/data/etl/transform/stainEstimator_gpu.py`

A `StainEstimatorGPU` class that mirrors `StainEstimator` but uses PyTorch:

```python
class StainEstimatorGPU:
    @staticmethod
    def get_profile(
        image: np.ndarray,
        method: str = 'macenko',
        device: str = 'cuda',
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    @staticmethod
    def _estimate_macenko(
        image: np.ndarray,
        Io: int = 240,
        alpha: float = 1,
        beta: float = 0.15,
        device: str = 'cuda',
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        ...
```

Uses lazy `import torch` inside methods (same convention as `ops/iou.py`).

**Key differences from NumPy version:**

| Operation | NumPy | PyTorch GPU |
|-----------|-------|-------------|
| `reshape + astype` | `np.reshape` | `torch.from_numpy(...).reshape()` |
| `log10` | `np.log10` | `torch.log10` |
| Covariance | `np.cov` | Manual: center + `torch.mm` |
| Eigendecomposition | `np.linalg.eigh` | `torch.linalg.eigh` |
| Percentile | `np.percentile` | `torch.quantile` |
| Pseudo-inverse | `np.linalg.pinv` | `torch.linalg.pinv` |
| Matrix multiply | `np.dot` | `torch.mm` or `@` |

**Fallback:** If CUDA is unavailable, auto-fallback to `StainEstimator` (NumPy).

### Phase 2: GPU Normalization

**File:** `raycasted/data/etl/transform/normalizer.py`

Add a `_apply_macenko_gpu` method to `NormalizerAndPadder`:

```python
def _apply_macenko_gpu(self, image: np.ndarray, Io: int = 240) -> np.ndarray:
    """GPU-accelerated Macenko normalization."""
    ...
```

The method:
1. Calls `StainEstimatorGPU._estimate_macenko(image)` on GPU.
2. Transfers the full image to GPU for the concentration→reconstruct pipeline.
3. Returns the normalized image as `np.ndarray`.

The heavy pixels-are-tensors steps (OD conversion, concentration computation,
reconstruction) all run on GPU. The target stain matrix and concentrations are
small (2×3 and (2,)) — kept on CPU or transferred once.

**Key consideration:** `_apply_macenko` in the current code calls
`StainEstimator._estimate_macenko` a second time per image (once for profiling
in Stage 2, once for normalization in Stage 3). The GPU path can optionally
cache the source stain matrix from Stage 2 to avoid redundant computation.
However, this is an optimization — the initial implementation can simply
re-estimate on GPU (still faster than NumPy).

### Phase 3: Orchestrator Integration

**File:** `raycasted/data/etl/transform/transform_orchestrator.py`

Add a `use_gpu` flag (auto-detected or config-driven):

```python
def __init__(self, config_manager, ingested_dir, final_output_dir):
    ...
    import torch
    self.use_gpu = torch.cuda.is_available()
```

**Stage 2 (`_build_population_profile`):**

```python
estimator = StainEstimatorGPU if self.use_gpu else StainEstimator
for row in self.registry.iter_rows(named=True):
    matrix, concentrations = estimator.get_profile(img, method='macenko')
    ...
```

**Stage 3 (`run_pipeline` normalization loop):**

```python
for row in self.registry.iter_rows(named=True):
    if self.use_gpu:
        final_img, final_annotations, content_h, content_w = self.normalizer.process_roi_gpu(img, annotations)
    else:
        final_img, final_annotations, content_h, content_w = self.normalizer.process_roi(img, annotations)
    ...
```

### Phase 4: Update Public API

**File:** `raycasted/data/etl/transform/__init__.py`

Add `StainEstimatorGPU` to exports.

### Phase 5: Tests

**New file:** `tests/phase_0_5/test_gpu_stain.py`

| Test | Description |
|------|-------------|
| `test_profile_matches_numpy` | Same image → GPU vs NumPy stain matrix, max element-wise error < 1e-4 |
| `test_concentrations_match_numpy` | Same image → GPU vs NumPy max concentrations, error < 1e-3 |
| `test_normalization_matches_numpy` | Full normalize pipeline → GPU vs NumPy output, pixel MAE < 1.0 |
| `test_white_image_fallback` | All-white image → returns (None, None) without error |
| `test_small_image` | 64×64 tile → produces valid profile |
| `test_batch_consistency` | 10 tiles → GPU profiles match NumPy profiles for each |

**Validation command:**

```bash
uv run python tests/phase_0_5/test_gpu_stain.py
```

---

## 5. Execution Order

```
1. Implement transform/stainEstimator_gpu.py             (Phase 1)
2. Write tests/phase_0_5/test_gpu_stain.py               (Phase 5)
3. Validate GPU stain estimator against NumPy baseline
4. Add GPU normalization path to normalizer.py            (Phase 2)
5. Update transform_orchestrator.py for GPU routing       (Phase 3)
6. Update transform/__init__.py                           (Phase 4)
7. Run full ETL pipeline test:
   uv run python -m raycasted.pipeline --config main/pannuke.yaml \
       --output output/gpu_transform_test --stage transform
8. Compare output .npz files against NumPy baseline
9. Run lint: uv run ruff check . && uv run ruff format .
```

---

## 6. What Stays on CPU

| Component | Reason |
|-----------|--------|
| `SpatialChunker` | Index arithmetic + array slicing. No heavy compute. |
| `filter_and_clip_annotations` | Operates on (N, 35) annotation arrays, not images. |
| `_pad_bottom_right` | `cv2.copyMakeBorder` — C++ optimized, trivial cost. |
| JSON I/O (profile read/write) | Negligible. |
| `.npz` load/save | I/O bound, not compute bound. |

---

## 7. Performance Expectations

### Per-Tile Cost

| Operation | NumPy (CPU) | PyTorch (GPU, Option A) |
|-----------|-------------|------------------------|
| OD conversion (1M pixels) | ~2ms | ~0.1ms |
| Cov + eigh (3×3) | ~0.5ms | ~0.05ms |
| Projection + percentile | ~1ms | ~0.1ms |
| pinv + concentrations | ~1ms | ~0.1ms |
| CPU↔GPU transfer | — | ~0.2ms |
| **Total per tile** | **~5ms** | **~0.6ms** |

### Full Pipeline (PanNuke ~2000 tiles)

| Stage | NumPy | GPU (Option A) |
|-------|-------|----------------|
| Stage 2: Profiling | ~10s | ~1.2s |
| Stage 3: Normalization | ~10s | ~1.2s |
| **Total transform** | **~20s** | **~2.5s** |

### VRAM Budget

Each 1024×1024×3 tile is ~12MB in float32. Processing one tile at a time (Option A)
requires ~50MB peak (input + OD + intermediate). No VRAM concern.

Option B (batched 16 tiles): ~800MB peak — still comfortable.

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `torch.quantile` vs `np.percentile` numerical differences | Tolerance in tests: element-wise error < 1e-3 for concentrations, pixel MAE < 1.0 for normalized images |
| Small 3×3 eigh/pinv: GPU launch overhead may dominate | Profile first. If overhead is too high, process on CPU with `torch` tensors (no `.cuda()`) — still benefits from PyTorch's optimized kernels without transfer |
| No CUDA available (CI/VM) | Auto-detect via `torch.cuda.is_available()`, fall back to NumPy path silently |
| Normalization changes pixel values → downstream training impact | Round-trip test validates pixel MAE < 1.0; if needed, increase tolerance after visual inspection |
| OD space `log10` underflow for zero pixels | Same `+1` guard as current implementation; `torch.log10` handles this identically |

---

## 9. Relationship to Ingestion GPU Refactor

This document is a companion to `docs/etl-gpu-raycast.md`. Together they cover
the full ETL GPU acceleration story:

```
Ingestion (etl-gpu-raycast.md):
    polygon_to_raycast → batch_polygon_to_raycast (GPU)
    ↓
    .npz files (unchanged format)
    ↓
Transform (this document):
    StainEstimator → StainEstimatorGPU (GPU)
    NormalizerAndPadder._apply_macenko → GPU path
    ↓
    .npz files (unchanged format)
    ↓
Loader / Training (no changes)
```

Both refactors share the same conventions:
- Lazy `import torch` inside function bodies (ETL safety).
- Auto-fallback to NumPy if CUDA unavailable.
- Old paths preserved, not removed.
- Tests validate GPU output against NumPy baseline.

---

## 10. Future Optimization: Batched GPU (Option B)

If profiling shows the GPU is underutilized with Option A (likely — per-tile
computation is too small to saturate a modern GPU), upgrade to Option B:

1. **Batch loader:** Load 8-16 tiles at a time into a `(B, H, W, 3)` GPU tensor.
2. **Masked covariance:** Compute per-image covariance using the diagonal mask
   trick (Section 3.3) to handle variable background pixel counts without
   fancy indexing.
3. **Batched eigh:** `torch.linalg.eigh` on `(B, 3, 3)` — single kernel launch.
4. **Batched pinv:** `torch.linalg.pinv` on `(B, 2, 3)` — single kernel launch.
5. **Concentration scaling + reconstruction:** All batched matrix multiplies.

This would reduce the per-tile overhead further by amortizing kernel launch
cost across 8-16 tiles. The implementation is more complex (masked operations,
variable-length handling) so it's deferred until Option A is validated.
