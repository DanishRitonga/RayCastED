# NuLite-DETR T12 deployment — ONNX / TensorRT on Jetson Orin Nano

Target model: the T12 control baseline
`output/v2/train52_fastvit-t12/weights/last.pt` (10.03M params, bPQ 0.6372 /
mPQ 0.4561 held-out Fold3, `main/pannuke-thin-t12.yaml`).

The NMS-free inference contract is preserved in deployment: the ONNX graph
contains only the raw decoder outputs, and the host script reproduces the
exact `eval_lsp` decode — argmax over `nc+1` with the no-object column
dropped, no confidence threshold, no NMS. The polygon-overlap ownership
resolution is evaluator-side (CPU/NumPy), as in the literature-facing
protocol.

## What was produced

| artifact | path |
|---|---|
| export script | `raycasted/export/export_nulite_detr.py` |
| deploy evaluator | `raycasted/evals/eval_lsp_deploy.py` |
| HF upload helper | `scripts/push_dataset_hf.py` |
| TRT build script | `scripts/build_trt_engine.sh` |
| exported graph | `output/v2/train52_fastvit-t12/deploy/train52_fastvit-t12.onnx` |
| metadata | `output/v2/train52_fastvit-t12/deploy/train52_fastvit-t12.meta.json` |

Graph contract (`dynamic batch`, fixed 256x256, opset 17):

- input `images`: `(B, 3, 256, 256)` float32, `[0,1]` (exactly what
  `RayCastTileDataset(augment=False)` emits — no extra normalization)
- output `dec_polygons`: `(B, 400, 66)` normalized polygons
- output `dec_logits`: `(B, 400, 6)` raw logits; column 5 = no-object

Validation against the PyTorch eval path: max abs diff 6.6e-7 (polygons) /
4.0e-5 (logits). 64-tile parity run vs `eval_lsp.py` (torch GPU) on ORT CPU:
identical bPQ 0.6415, bDQ 0.7922, mPQ 0.5061, F1 0.7874 (mSQ differs 0.0001 —
fp32 logit noise on one tie-boundary pixel).

The historical `raycasted/export/onnx_export.py` + `raycasted/evals/eval_onnx.py`
target the old YOLO26-FCN RayCastED and are untouched.

## Workflow on the training machine (once)

1. Export (already done for T12):

   ```bash
   uv run python -m raycasted.export.export_nulite_detr \
       --weights output/v2/train52_fastvit-t12/weights/last.pt --validate
   ```

2. Push the transformed splits to HuggingFace (train/val/test, 7,901 tiles):

   ```bash
   HF_REPO_ID=<user>/<repo> uv run python scripts/push_dataset_hf.py
   ```

   Add `--with-parquet` to also upload the raw fold parquets (only needed if
   the device must generate tiles locally with the ETL pipeline).

## Workflow on the Jetson (Orin Nano)

Pull the repo (the deploy paths only need `raycasted/` + `scripts/`; the
checkpoint .pt is NOT needed on the device), copy or fetch
`deploy/train52_fastvit-t12.onnx` + `.meta.json`.

Dependencies: `pip install huggingface_hub polars pyarrow opencv-python-headless scipy pillow onnxruntime`
(TRT path additionally: `pycuda`, `tensorrt` — both ship with JetPack).

1. Build the engine on-device (engines are device-specific — never copy an
   engine from x86):

   ```bash
   BATCH=8 scripts/build_trt_engine.sh deploy/train52_fastvit-t12.onnx
   # -> deploy/train52_fastvit-t12.fp16.engine
   ```

2. Evaluate the held-out fold straight from HuggingFace:

   ```bash
   # TensorRT:
   python -m raycasted.evals.eval_lsp_deploy \
       --engine deploy/train52_fastvit-t12.fp16.engine \
       --meta   deploy/train52_fastvit-t12.meta.json \
       --hf-dataset <user>/pannuke-fold3-test \
       --batch 8 --workers 2

   # Or, without TensorRT (ONNX Runtime; CUDA EP if the wheel has it):
   python -m raycasted.evals.eval_lsp_deploy \
       --onnx deploy/train52_fastvit-t12.onnx \
       --meta deploy/train52_fastvit-t12.meta.json \
       --hf-dataset <user>/pannuke-fold3-test \
       --batch 8 --workers 2
   ```

   Local copies of `--data-dir`/`--parquet` override the HF fetch when given;
   `--max-images N` runs the first N sorted tiles for a quick smoke test;
   `--cpu` forces the ORT CPU EP.

3. Optional parity/audit on the Jetson:

   ```bash
   # ONNX backend only: save raw predictions for audit_lsp_cohorts.py
   python -m raycasted.evals.eval_lsp_deploy --onnx ... --save-predictions preds.npz ...
   ```

Notes:

- `--save-predictions` is ORT-only (the TRT path keeps all results in memory
  the same way; the restriction is intentional, see the script docstring).
- Expected Jetson latency: measure from the `Inference: X ms/img` line the
  script prints; post-processing (PIL rasterization + EDT overlap) runs on
  CPU and dominates wall time on a full 2,722-tile fold.
- Numbers are literature-comparable ONLY against the `eval_lsp` protocol on
  the genuinely held-out fold (fold3 for this checkpoint).
