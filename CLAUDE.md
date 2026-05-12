# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
uv sync

# Lint + format (always run together before committing)
uv run ruff check . && uv run ruff format .

# Run all tests for a phase
uv run python -m pytest tests/phase_6/ -x -q

# Run a single test file
uv run python -m pytest tests/phase_6/test_loss.py -x -q

# Run a single test function
uv run python -m pytest tests/phase_6/test_loss.py::test_constructor -xvs

# Full pipeline
uv run python -m raycasted.pipeline --config main/pannuke.yaml --output output/<RunName> --stage train --epochs 500 --device 0

# Individual stages
uv run python -m raycasted.pipeline --config main/dataset.yaml --output output/ --stage ingest --dataset PUMA
uv run python -m raycasted.pipeline --config main/dataset.yaml --output output/ --stage transform

# Clean caches (run after tests)
bash clean_cache.sh
```

**Do NOT launch training commands.** Update `main/pannuke.yaml` and tell the user to run training themselves.

## Architecture

RayCastED converts YOLOv26 from bounding-box detection to raycast polygon detection for cell detection in histopathology WSIs.

### Data flow

```
Raw datasets (Parquet/GeoJSON/CSV)
  → IngestionOrchestrator → .npz files (image + raycast annotations, pixel space)
  → TransformOrchestrator (SpatialChunker + NormalizerAndPadder) → .npz tiles
  → RayCastTileDataset → [B, 3, H, W] + [M, 36] labels (normalised by crop_size)
  → RayCastTrainer (RayCastE2ELoss + RayCastAssigner) → trained weights
  → RayCastPredictor → [N, R, 2] polygon vertices
```

### Annotation format

Single array format across all stages — no conversion between ETL and model:
```
[class_id, cx, cy, d_1..d_R]   shape: (N, 2+R+1), float32, pixel space
```
Collated batch adds leading `batch_idx`: `(sum_M, 2+R+2)`. Normalisation (divide by crop_size) happens **only** in `RayCastTileDataset._normalise()`.

### Key modules

- `raycasted/data/etl/ops/` — single source of truth for ALL geometry (PyTorch variants use lazy imports)
- `raycasted/data/etl/utils/constants.py` — angular convention, format indices (`RAY_START_IDX`, etc.)
- `raycasted/model/blocks/head.py` — `RayCastDetect` head (subclasses `Detect`, replaces bbox with raycast)
- `raycasted/model/loss.py` — `RayCastE2ELoss` (4-term polygon loss: xy/cls/l1/smooth + aux_xy)
- `raycasted/model/tal.py` — `RayCastAssigner` (Polar-IoU matching) + `HungarianRayCastAssigner` (one2one)
- `raycasted/model/train.py` — `RayCastTrainer` (subclasses `DetectionTrainer`)
- `raycasted/model/builder.py` — `raycasted_parse_model()` (custom YAML parser for ResoConv, C3k2_LK blocks)
- `raycasted/pipeline.py` — CLI orchestrator chaining ingest → transform → train

### E2E dual-assignment (NMS-free)

Uses `E2ELoss` with two branches:
- **one2many**: `RayCastAssigner` with `topk=tal_topk`, provides dense supervision scaffold
- **one2one**: `HungarianRayCastAssigner` with `topk=1`, enables NMS-free inference
- `o2m` weight decays from 0.8 → 0.1 over training, transferring learning to the one2one branch
- **topk2 matters**: one2one uses `topk=max(tal_topk//2, 7), topk2=1` (candidate pool then filter to 1). Never set `topk=1` directly — it skips the candidate pool.
- One2many uses `topk2=topk` to disable secondary filtering.

### Training curriculum (from-scratch, no pretrained weights)

- Epochs 0–50: Assigner warmup uses Gaussian centroid-distance (not Polar-IoU) because random backbone features produce meaningless ray predictions. `lambda_l1=0.1` (near-zero) keeps ray params alive for DDP.
- Epochs 50+: Assigner switches to Polar-IoU, `lambda_l1` jumps to 25.0.
- Smooth loss anneals 0 → 1 over first 40% of training.
- Aux XY head (1×1 conv on neck features) provides direct backbone→centroid gradient, decays after 80% of training.

## Code Style

Ruff config: line-length 120, single quotes, Google-style docstrings, isort with `raycasted` as first-party.

## Key Gotchas

- **Parallel processing**: Use `multiprocessing.get_context('spawn')` — `fork` deadlocks with OpenMP-backed libraries.
- **AMP**: Cast to `.float()` before `torch.cdist` and IoU computations (float16 underflows).
- **Validator**: `RayCastTileDataset` returns float32 [0,1] images. Must override `preprocess_batch` to skip /255.
- **InfiniteDataLoader**: Must use Ultralytics' `InfiniteDataLoader` (not plain `DataLoader`) — Ultralytics calls `train_loader.reset()`.
- **XY decode**: Training and inference must use `(sigmoid * 2.0 - 0.5 + anchor) * stride`.
- **Bias init**: `RayCastDetect.bias_init()` must use `crop_size` (not stride) for normalised-space predictions.
- **Loss is a 4-element tensor**: `RayCastDetectionLoss.loss()` returns `(loss * batch_size, loss_detach)` where `loss` is `[xy, cls, l1, smooth]`. Ultralytics calls `.sum()` on the result for backward. When adding new loss terms, append as new elements (don't add scalars — they broadcast).
- **n_rays is mutable**: `constants.py` has `configure_rays(n)` that mutates module-level `N_RAYS`. Pipeline calls it at startup; tests must call it too.
- **No pretrained weights**: This is a new architecture. COCO pretrained backbone is counterproductive due to domain gap (natural images → H&E histopathology).

## Spec

Authoritative specification: `docs/project.md` (v4.17). Read it before modifying any module.
