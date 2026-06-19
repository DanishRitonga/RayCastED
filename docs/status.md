# RayCastED — Current Status

> Last updated: 2026-04-12

## Evaluation Metrics

PanNuke fold3 evaluation uses `raycasted/scripts/eval_pannuke.py`, which computes comprehensive pixel-level and detection metrics using **mask IoU** (not polar IoU):

| Metric | Description | Method |
|--------|-------------|--------|
| **AJI** | Aggregated Jaccard Index | Hungarian matching at IoU≥0.5, sums intersection/union over all matched+unmatched |
| **mAP@0.5** | Mean Average Precision at IoU=0.5 | Mask IoU with greedy matching, per-class AP via all-point interpolation |
| **mAP@0.75** | Mean Average Precision at IoU=0.75 | Same as above at stricter threshold |
| **mAP@0.5:0.95** | Mean AP over IoU thresholds 0.50–0.95 (step 0.05) | COCO-style averaged mAP |
| **PQ** | Panoptic Quality = SQ × DQ | Hungarian matching at IoU≥0.5 |
| **SQ** | Segmentation Quality | Mean IoU of matched pairs (shape accuracy) |
| **DQ** | Detection Quality | TP / (TP + 0.5×FP + 0.5×FN), equivalent to F1 |
| **Precision** | Per-pixel precision at IoU=0.5 | TP / (TP + FP) aggregated across all classes |
| **Recall** | Per-pixel recall at IoU=0.5 | TP / (TP + FN) aggregated across all classes |
| **F1** | Harmonic mean of Precision and Recall | 2×P×R / (P+R) |
| **Params** | Model parameters (M) | Total parameter count |
| **GFLOPs** | Giga floating-point operations | Via `model_info()` at crop_size resolution |
| **Inference Time** | ms per image (batch=1) | Warmup 10 batches, then average |

Key differences from training-time metrics:
- **Training validator** (`RayCastValidator`): uses GPU **Polar IoU** (shape-only, no centroid)
- **PanNuke eval** (`eval_pannuke.py`): uses **Mask IoU** (rasterized polygons, accounts for centroid + shape)
- Predicted masks use **overlap resolution** (largest-first priority), matching LSP-DETR's post-processing

```bash
# Run evaluation
uv run python -m raycasted.scripts.eval_pannuke \
    --weights train4/weights/best.pt \
    --data-dir output/pannuke/transformed/test \
    --batch 16 --device 0 --conf 0.25
```

## Training Results

PanNuke, 175 epochs, crop_size=256, batch=16, YOLOv26s backbone:

| Metric | Value |
|--------|-------|
| mAP50(P) | 0.56 |
| mAP50-95(P) | 0.40 |
| Precision | 0.606 |
| Recall | 0.553 |
| Train cls_loss | 3.09 → 0.33 |
| Train piou_loss | 0.82 → 0.25 |

Training losses converge steadily. Metrics plateau around epoch 50 and remain stable through epoch 175 — no collapse.

PUMA (202 ROIs) also trains successfully: 2 GPU epochs in 174s end-to-end.

## Bugs Fixed

| Bug | File | Fix |
|-----|------|-----|
| **Validator double normalization (mAP collapse)** | `val.py:160-173` | Override `preprocess` to skip `/255` — `RayCastTileDataset` already returns [0,1] images. Double-normalising produced ~0.004 values, causing mAP to collapse from 0.11 → 0. |
| **XY decode mismatch (training vs inference)** | `loss.py:99` | Training `decode_pred_xy` now uses `(sigmoid * 2.0 - 0.5 + anchor) * stride` to match the inference decode in `head.py:126`. Previously training used `(anchor + sigmoid) * stride`, causing a systematic centroid shift. |
| **AMP dtype mismatch (assigner)** | `tal.py:131` | Cast both `gt_xy` and `xy_centers` to `.float()` before `torch.cdist` — float16/float32 mix under AMP autocast. |
| **AMP dtype mismatch (IoU)** | `tal.py:210-212` | `.to(overlaps.dtype)` before assignment — float16 IoU tensor into float32 output. |
| **Ignore class reaching loss** | `raycast_dataset.py:60-62` | Filter `class_id=255` annotations in `__getitem__` — caused index-out-of-bounds in assigner with nc=5. |
| **`nt_per_class` NoneType on epoch 1** | `val.py:50-60` | Compute `nt_per_class` from `target_cls` before early return in `RayCastDetMetrics.process()`. |
| **`cls.shape[0]` IndexError** | `val.py:264` | `.flatten()` instead of `.squeeze(-1)` — single-element tensor squeeze produces 0-dim scalar. |
| **`DataLoader.reset()` AttributeError** | `train.py:26,234` | Use `InfiniteDataLoader` — Ultralytics' `_do_train` calls `self.train_loader.reset()` after training. |
| **`plot_images`/`plot_predictions` crash** | `train.py:149`, `val.py:348` | Override as no-ops — Ultralytics expects 4-dim bboxes, crashes on 34-dim polygon data. |
| **`plot_val_samples` crash** | `val.py:367` | Override as no-op — same xywh2xyxy assertion error as plot_predictions, triggered during validation plotting. |
| **`confusion_matrix.plot()` AttributeError** | `val.py:169-170` | Override `finalize_metrics` as no-op — `self.confusion_matrix = None`. |
| **Parallel ingestion deadlock** | `ingestion_orchestrator.py:199`, `parquet_ingestor.py:223` | Use `multiprocessing.get_context('spawn')` — default `fork` deadlocks with OpenMP-backed libraries. |
| **PanNuke parallelism (few parquet files)** | `parquet_ingestor.py` | Internal ROI-level parallelism — reads parquet in main process, dispatches per-ROI tasks to workers. |
| **Bias init in stride-space** | `head.py:155`, `train.py:277` | Compute as `log(exp(target_px / crop_size) - 1)` in normalised space, not `target_px / stride`. |
| **`output_image_size`/`max_size` unification** | `config.py:36-53` | Two validated Pydantic fields: `max_size` (ETL tile size) and `crop_size` (model input size). |

## Known Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| **Val losses all NaN** | **Fixed** | Caused by XY decode mismatch (training used `(anchor + sigmoid) * stride` while inference used `(sigmoid * 2.0 - 0.5 + anchor) * stride`). Garbage xy during validation produced NaN in polar IoU. Fixed alongside the XY decode mismatch in `loss.py:99`. |
| **`MatInstIngestor._extract_raycast_annotations` is a stub** | Low | `NotImplementedError` — deferred, requires contour extraction from instance maps. |
| **`GeoJSONIngestor` fails on 4 PUMA ROIs** | Low | `'LineString' object has no attribute 'x'` — degenerate geometries not handled. |
| **`plot_training_labels` is no-op** | Low | Standard bbox plotting incompatible with 34-dim data. Custom polygon plotting deferred. |
| **MLflow annealing logging needs custom callback** | Low | `lambda_smooth`/`o2m_weight` not logged per epoch; per-term losses auto-logged by Ultralytics. |
| **Ingestor diagnostic counters not wired** | Low | `fallback_counter` not passed to `polygon_to_raycast()`, zero-ray logging not implemented. |

## What's Working

- **ETL pipeline**: All 3 ingestors (GeoJSON, CSV, Parquet) with raycast extraction
- **Transform pipeline**: SpatialChunker + NormalizerAndPadder with content_h/w
- **DataLoader**: RayCastTileDataset with crop/augment/normalise, InfiniteDataLoader
- **Model head**: RayCastDetect(Detect) with RayRefinementBlock, registered via `register_raycast_head()`
- **Loss**: 5-term RayCastE2ELoss with smoothness annealing (0.05 → 0 over 50 epochs)
- **Assigner**: RayCastAssigner with VRAM-safe Polar-IoU, 75th-percentile radius containment
- **Validator**: GPU Polar-IoU mAP + centroid F1, no NMS (end-to-end Hungarian)
- **Trainer**: RayCastTrainer wired into Ultralytics loop, checkpoint save/resume with metadata
- **Pipeline orchestrator**: End-to-end CLI (`ingest → transform → train`)
- **Parallel ingestion**: `spawn`-based ProcessPoolExecutor, internal ROI-level parallelism for Parquet

## What's Not Yet Implemented

- **Phase 9**: ONNX export + TensorRT deployment (separate hardware — Jetson)
- **MLflow custom callback** for annealing schedule logging
- **Custom polygon label plotting** (currently no-op)
- **Ingestor diagnostic counters** (fallback rate, zero-ray fraction)

## Key Architecture Decisions

1. **Subclass, not fork**: All ultralytics modifications are subclasses under `raycasted/model/`, not in-place edits to `ultralytics/`.
2. **End-to-end (NMS-free)**: `RayCastDetect` uses `end2end=True` with dual assignment (one2many for training signal, one2one for inference).
3. **Unified annotation format**: `[class_id, cx, cy, d_1..d_32]` (shape 35) used at all stages — no conversion between ETL and model.
4. **Normalisation in DataLoader only**: `RayCastTileDataset._normalise()` divides by `crop_size`. All other stages work in pixel space.
5. **DFL removed**: Replaced with polygon regression (34-dim output). `self.dfl = nn.Identity()`.
