# RayCastED — Agent Notes

Compact instruction file for future sessions. Read `CLAUDE.md` for full commands and architecture; this file covers what's easy to miss.

## Quick Reference

```bash
uv sync                                          # install
uv run ruff check . && uv run ruff format .       # lint + format (always before commit)
uv run python -m pytest tests/phase_6/ -x -q     # main test suite (27 tests)
uv run python -m pytest tests/phase_6/test_loss.py::test_constructor -xvs  # single test
bash clean_cache.sh                               # clean __pycache__ after tests
```

**Never launch training.** Edit `main/pannuke.yaml` and tell the user to run it.

## What This Project Is

RayCastED is a YOLOv26 variant that replaces bounding-box detection with **raycast polygon detection** for cell nuclei in histopathology WSIs. Each detection outputs a centroid + 64 radial rays (not axis-aligned boxes). No pretrained weights — trained from scratch on H&E tissue images.

## Architecture That's Not Obvious From Filenames

- `raycasted/model/` is a **subclass package** — all Ultralytics modifications extend base classes in-place, never editing `ultralytics/` source
- `raycasted/model/train.py` contains `_RayCastCriterionWrapper` which threads YAML config → loss/assigner construction. Any new loss param must pass through: YAML → `train.py` wrapper → `RayCastE2ELoss.__init__` → `RayCastDetectionLoss.__init__` → assigner constructors
- `raycasted/model/builder.py` has `raycasted_parse_model()` — custom YAML parser for `ResoConv`, `C3k2_LK`, `ResoConvHybrid` blocks not in standard Ultralytics
- `raycasted/data/etl/utils/constants.py` has **mutable module-level globals** (`N_RAYS`, `RAY_ANGLES`, etc.) changed via `configure_rays(n)`. Pipeline calls it at startup; tests must call it too
- The P2-P3-P4 pyramid (stride 4/8/16) is defined in `raycasted/cfg/yolo26s-run28-p234.yaml`, not the standard P3-P4-P5

## Training History — What We've Tried and Why

This project went through extensive trial-and-error. Understanding what was tried prevents repeating mistakes:

### Removed: Assignment Warmup / Curriculum Learning
The codebase previously had a 50-epoch "warmup" where the assigner used Gaussian centroid-distance similarity instead of Polar-IoU, combined with lambda annealing that crushed L1/piou losses to 0.1 during warmup. This was **removed** because:
- Tight warmup sigma puts assigned anchors in the L2 regime of Huber loss → tiny gradients → xy loss stuck at ~8.4 for 10+ epochs
- Crushing lambda_l1 to 0.1 prevented ray learning for 50 epochs — predictions were still random when the ramp started
- The curriculum was fighting itself: the assignment already handles curriculum by ranking on centroids; the loss weights didn't need to also suppress rays

### Removed: Hungarian Matching for o2o Branch
Originally used `HungarianRayCastAssigner` (scipy `linear_sum_assignment`) for the one2one branch. Replaced with **standard dual-TAL** (same `RayCastAssigner` for both branches, different topk) because:
- At epoch 0, predicted rays are random noise → Hungarian cost matrix is noise-dominated → confident-but-wrong 1:1 matches
- No soft quality scores to downweight bad matches (unlike TAL which has alignment metrics)
- Stock YOLO26 dual-TAL is simpler and more robust: o2m topk=15/topk2=15, o2o topk=7/topk2=1

### Current Architecture: Standard Dual-TAL + PLB + bg_cls_decay
- Both branches use `RayCastAssigner` with Polar-IoU from epoch 0
- PLB (Pixel-Level Balancing): area-based fg weighting `2*(1 - area/total_area)` boosts small nuclei
- bg_cls_decay: downweights bg anchor cls loss
- Static lambdas (no annealing): lambda_l1=25.0, lambda_piou=2.0

### Classification Mode Collapse — The Gradient Budget Problem
The model achieved Recall=0.881 (excellent localization) but Precision=0.074 (catastrophic classification). Root cause is a gradient budget imbalance:

With focal_alpha=0.25, bg_fg_ratio=3, bg_cls_decay=0.5:
- Per-anchor: fg=0.031, bg=0.094 → 3:1 bg:fg
- Count ratio: 3:1 bg:fg
- Total gradient: **4.5:1 bg:fg** — 82% of cls gradient budget goes to "be background"

This leaves only 18% for inter-class discrimination across 5 classes with wildly different frequencies (Neo=40.8%, Dead=1.5%). Dead class gets ~0.3% of total gradient.

**What didn't work:**
- focal_alpha=0.5 → removed bg dominance but also removed focal's easy-negative suppression → different mode collapse
- class_weights → amplified rare-class fg, but with bg still dominant at 4.5:1, the amplified signal was still too weak → model overfit to predicting the least-penalized class everywhere
- soft_targets → quality scores prevented convergence in E2E dual-assigner setup

**What should work (not yet applied):**
- focal_alpha=0.75 (flip fg/bg weights) → per-anchor 3:1 fg:bg, with bg_fg_ratio=3 → total 1:1 balance
- Then class_weights with cap at 2.0 becomes safe because fg signal is strong enough

## Loss System

5-term tensor: `[xy, cls, L1, piou, smooth]` plus optional `aux_xy`. When adding new loss terms, append as new tensor elements — never add scalars (they broadcast incorrectly).

- **Loss config threading**: `pannuke.yaml` → `_RayCastCriterionWrapper.__call__` → `RayCastE2ELoss.__init__` → `RayCastDetectionLoss.__init__` → assigners
- **PLB** (Pixel-Level Balancing): area-based fg weighting `2*(1 - area/total_area)` boosts small nuclei. Controlled by `plb_enabled` in YAML
- **bg_cls_decay**: downweights bg anchor cls loss. Applied once in shared path — do NOT reapply in o2o branch
- **E2E dual-assignment**: Both branches use `RayCastAssigner` (dual-TAL). o2m: topk=15, topk2=15. o2o: topk=7, topk2=1 (NMS-free). `o2m` weight decays 0.8→0.1 over training
- **Static lambdas**: lambda_l1=25.0, lambda_piou=2.0 (no ramp, no warmup). Smooth annealing still exists (0→1 over 40% training)

## P2 Background Anchor Flood

With P2-P3-P4 pyramid, P2 contributes ~4,096 out of 5,376 anchors (76%). Many P2 anchors are near nucleus boundaries and produce noisy cls gradients. The `bg_fg_ratio=3` subsampling helps but doesn't fully solve it. Per-level normalization or spatial bg masking are documented approaches but not yet implemented.

## Critical Gotchas

- **Validator double-normalization**: `RayCastTileDataset` returns float32 [0,1] images. `RayCastValidator.preprocess` must skip /255 — double-normalizing collapses mAP to 0
- **XY decode**: Both training and inference must use `(sigmoid * 2.0 - 0.5 + anchor) * stride` — mismatch causes NaN losses
- **AMP**: Cast to `.float()` before `torch.cdist` and IoU computations — float16 underflows
- **InfiniteDataLoader**: Must use Ultralytics' version, not plain `DataLoader` — Ultralytics calls `train_loader.reset()`
- **multiprocessing**: Use `get_context('spawn')` — default `fork` deadlocks with OpenMP
- **Bias init**: `RayCastDetect.bias_init()` uses `crop_size` (not stride) for normalised-space predictions. Formula: `log(exp(target_px / crop_size) - 1)` in normalised space
- **nc propagation**: `nc` must be passed at model construction time (`get_model()`), not patched after — the cv3 conv layers are built with wrong channel count otherwise. An assertion guard exists: `head.nc == self.model.nc`
- **n_rays is mutable**: `constants.py` module-level state. Always call `configure_rays(n)` at startup
- **Fallback strides**: Loss and assigner had hardcoded `[8,16,32]` fallback — wrong for P2-P3-P4 architecture (should be `[4,8,16]`). Check if still present when debugging stride-related issues
- **loss_detach**: Must use o2o branch (not o2m) for progress bar display — otherwise logs wrong branch's metrics
- **Huber delta**: Currently 0.05 for xy loss. If xy loss gets stuck with small raw values, the delta may be putting errors in L2 regime with tiny gradients. Consider increasing to 0.1 or using pure L1
- **target_scores_sum**: Uses binarized cls targets (hard 0/1), not soft alignment scores. Soft scores inflated the denominator ~15%, weakening cls loss

## Diagnostic Infrastructure

The loss has `DIAG` logging that prints per-branch raw loss values at regular step intervals:
```
DIAG o2m step=100 | fg=3293/86016 | raw: xy=0.0004 cls=2.79 l1=0.49 piou=0.95 smooth=0.007
DIAG o2o step=100 | fg=377/86016 | raw: xy=0.016 cls=3.91 l1=0.51 piou=0.94 smooth=0.007
```
- `fg=X/Y` shows foreground anchors / total — if fg is very low, the assigner isn't matching
- `raw` values are pre-lambda — multiply by lambda to get actual contribution
- Reported `xy_loss` = raw_xy × lambda_xy (500) — so raw_xy=0.017 → reported 8.5

## Key Files

| File | Purpose |
|------|---------|
| `raycasted/model/loss.py` | 5-term polygon loss + PLB + bg decay |
| `raycasted/model/tal.py` | RayCastAssigner (Polar-IoU, dual-TAL) |
| `raycasted/model/train.py` | RayCastTrainer + config wiring |
| `raycasted/model/blocks/head.py` | RayCastDetect head (replaces bbox with raycast) |
| `raycasted/model/builder.py` | Custom YAML model parser |
| `raycasted/pipeline.py` | CLI orchestrator (ingest → transform → train) |
| `raycasted/data/etl/utils/constants.py` | Mutable ray geometry constants |
| `main/pannuke.yaml` | Training config (all hyperparams live here) |
| `docs/project.md` | Authoritative spec (v4.17, 2200+ lines) |

## Spec & Docs

- `docs/project.md` is the authoritative specification — read it before modifying any module
- `CLAUDE.md` has the full command reference and architecture overview
- `docs/status.md` tracks bugs fixed, known issues, and training results
