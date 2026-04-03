# RayCastED

**RayCast-based End-to-end Detection** — *ray-cast* it.

A raycast polygon detector built on YOLOv26 for detecting cells in histopathological whole-slide images. Instead of axis-aligned bounding boxes, RayCastED predicts each cell as a centroid + 32 radial rays, producing a polygon boundary that captures real cell morphology.

## Pipeline

```
Raw datasets (Parquet / GeoJSON / CSV)
        ↓  IngestionOrchestrator
.npz files  [image + raycast annotations, pixel space]
        ↓  TransformOrchestrator (SpatialChunker + NormalizerAndPadder)
.npz tiles  [content_h, content_w preserved]
        ↓  PolygonTileDataset
[B, 3, H, W] + [M, 36] labels (normalised)
        ↓  RayCastED
Trained weights
        ↓  PolygonPredictor
[N, 32, 2] polygon vertices (pixel space)
```

## Quick start

```bash
# Install dependencies (requires uv + Python 3.11)
uv sync

# Run round-trip geometry tests
uv run python tests/phase_0_5/test_round_trip.py
```

## Documentation

The authoritative specification is [`docs/project.md`](docs/project.md) — read it before modifying any module.

## Datasets

| Dataset | Format | Cells |
|---------|--------|-------|
| MoNuSAC | Parquet (instance masks) | Multi-organ |
| PUMA | GeoJSON (polygon vertices) | Multi-organ |
| PanopTILs | CSV (polygon coordinates) | TILs |

## Deployment

Target: NVIDIA Jetson (Orin/Xavier) via ONNX → TensorRT FP16 inference.
