# ONNX & TensorRT Deployment

Export RayCastED checkpoints to ONNX and deploy on NVIDIA Jetson (Orin Nano)
with TensorRT for accelerated inference.

## Prerequisites

- Python 3.10+ (3.10 required on Jetson for system TensorRT bindings)
- CUDA toolkit matching the driver version
- TensorRT (system-installed on Jetson; pip-installable on desktop GPU)

## Step 1: Export to ONNX

```bash
uv run python -m raycasted.export.onnx_export \
    --weights docs/runs/v1.1/best.pt \
    --output export/raycast_model.onnx \
    --validate
```

Output files:

- `export/raycast_model.onnx` — static graph with raw head logits
- `export/raycast_model.meta.json` — geometry metadata (n_rays, strides,
  conf_threshold, ray_cos/sin, etc.)

The export wrapper strips one2many heads, applies inter-scale competition
inside the graph (sigmoid → ISC → clamp → log for logit-to-logit round-trip),
and returns raw logits without activations or decoding.

Validation checks PyTorch vs ONNX numerical equivalence (default tolerance
`atol=1e-4, rtol=1e-3`; passes at ~5e-5 max diff).

## Step 2: Build TensorRT Engine

### On Desktop GPU (pip tensorrt)

```bash
uv run python -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)
with open('export/raycast_model.onnx', 'rb') as f: parser.parse(f.read())
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
engine = builder.build_serialized_network(network, config)
with open('export/raycast_model.engine', 'wb') as f: f.write(engine)
print('Engine built')
"
```

### On Jetson (trtexec)

```bash
# FP32
/usr/src/tensorrt/bin/trtexec \
    --onnx=export/raycast_model.onnx \
    --saveEngine=export/raycast_model.engine \
    --memPoolSize=workspace:1024

# FP16 (2x faster, marginal quality loss)
/usr/src/tensorrt/bin/trtexec \
    --onnx=export/raycast_model.onnx \
    --saveEngine=export/raycast_model.engine \
    --fp16 \
    --memPoolSize=workspace:1024

# FP16 + CUDA graphs (reduces kernel-launch overhead)
/usr/src/tensorrt/bin/trtexec \
    --onnx=export/raycast_model.onnx \
    --saveEngine=export/raycast_model.engine \
    --fp16 \
    --useCudaGraph \
    --memPoolSize=workspace:1024
```

## Step 3: Evaluate

### Desktop (full test set)

```bash
uv run python main/eval_onnx.py \
    --engine export/raycast_model.engine \
    --meta export/raycast_model.meta.json \
    --data-dir /path/to/test_tiles \
    --conf 0.5
```

Metrics: AJI, AP@0.5/0.5:0.95, bPQ, mPQ, F1, precision, recall,
inference time.

### Jetson Orin Nano

```bash
uv run --no-sync python main/eval_jetson.py \
    --engine ./onnx/v1.engine \
    --meta ./onnx/v1.meta.json \
    --data-dir /path/to/test_tiles \
    --conf 0.5
```

Self-contained — no PyTorch, ultralytics, or raycasted package. Only numpy,
scipy, opencv-python, pycuda, and system tensorrt. Uses the same
`compute_metrics_streaming` implementation as `eval_pannuke.py`.

### Visual Test (per-sample polygon overlays)

```bash
# PyTorch (desktop)
uv run python main/visualize_predictions.py --weights docs/runs/v1.1/best.pt \
    --data-dir output/visualTest --conf 0.49

# ONNX Runtime (any platform)
uv run python main/visualize_predictions.py --onnx export/raycast_model.onnx \
    --meta export/raycast_model.meta.json --data-dir output/visualTest --conf 0.49

# TensorRT (desktop or Jetson)
uv run python main/visualize_predictions.py --engine export/raycast_model.engine \
    --meta export/raycast_model.meta.json --data-dir output/visualTest --conf 0.49
```

Output per sample: `{data-dir}/output/{filename}/gt.png | pred.png | combined.png`

## Jetson Setup Notes

### Dependencies

```bash
# Build pycuda against the correct CUDA
CPLUS_INCLUDE_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/include \
CUDA_ROOT=/usr/local/cuda-12.6 \
LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib \
LDFLAGS="-L/usr/local/cuda-12.6/targets/aarch64-linux/lib/stubs" \
uv pip install pycuda

# Use system TensorRT (Python 3.10 only on JetPack 6.x)
ln -s /usr/lib/python3.10/dist-packages/tensorrt .venv/lib/python3.10/site-packages/

# Lightweight deps for eval_jetson.py (no torch/ultralytics)
uv pip install numpy scipy opencv-python
```

### Known Issues

- TensorRT graph optimization (layer fusion, kernel selection) changes FP32
  numerical accumulation order. Expect ±0.01 metric difference vs PyTorch
  even in FP32 mode. This is not a bug — it is expected TensorRT behavior.
- ONNX Runtime CPU inference matches PyTorch within `5e-6` max diff.
- Jetson Orin Nano is ~38ms/img FP32, ~19ms FP16 for 256×256 input.
- Desktop GPU reaches 2.7ms/img for the same model.
