"""RayCastED — Jetson Inference Runtime (Phase 9).

Loads a TensorRT engine, runs inference on WSI tiles, and post-processes
raw head output into polygon detections. Runs on NVIDIA Jetson (Orin/Xavier).

Usage:
    runtime = JetsonRuntime('polygon_yolo.engine', 'polygon_yolo_meta.json')
    polygons = runtime.infer(tile_image)
    # polygons: [N, 36] = [cx, cy, d_1..d_32, score, cls_idx]

Requires: TensorRT Python bindings (installed with JetPack).
"""

import json

import numpy as np

from raycasted.export.postprocess import postprocess_raw_output


class JetsonRuntime:
    """TensorRT inference runtime for RayCastED on Jetson.

    Handles engine loading, memory allocation, preprocessing,
    inference execution, and polygon post-processing.

    Args:
        engine_path: Path to TensorRT .engine file.
        meta_path: Path to metadata JSON sidecar.
    """

    def __init__(self, engine_path: str, meta_path: str):
        self.meta = self._load_meta(meta_path)
        self.imgsz = self.meta['imgsz']
        self.conf_threshold = self.meta.get('conf_threshold', 0.25)
        self.dedup_radius = self.meta.get('dedup_radius_px', 5.0)
        self.strides = self.meta.get('strides', [8, 16, 32])

        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._setup_buffers()

    @staticmethod
    def _load_meta(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def _load_engine(self, path: str):
        """Load TensorRT engine from file.

        Requires tensorrt package (available on Jetson via JetPack).
        """
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(path, 'rb') as f:
            return runtime.deserialize_cuda_engine(f.read())

    def _setup_buffers(self):
        """Allocate host and device memory for input/output tensors."""
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt_dtype_to_np(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorMode.INPUT:
                self.input_name = name
                self.input_host = host_mem
                self.input_device = device_mem
            else:
                self.output_name = name
                self.output_host = host_mem
                self.output_device = device_mem

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        Resize to imgsz, normalise [0,1], convert to CHW float32.

        Args:
            image: [H, W, 3] uint8 BGR image.

        Returns:
            [1, 3, imgsz, imgsz] float32 tensor.
        """
        import cv2

        resized = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
        return np.ascontiguousarray(blob)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image.

        Args:
            image: [H, W, 3] uint8 BGR image.

        Returns:
            [N, 36] polygon detections: [cx, cy, d_1..d_32, score, cls_idx].
        """
        import pycuda.driver as cuda

        blob = self.preprocess(image)
        np.copyto(self.input_host, blob.ravel())

        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()

        raw = self.output_host.reshape(1, -1, self.meta['nc'] + 34)
        results = postprocess_raw_output(
            raw,
            strides=self.strides,
            imgsz=self.imgsz,
            conf_threshold=self.conf_threshold,
            dedup_radius_px=self.dedup_radius,
        )
        return results[0]


def trt_dtype_to_np(trt_dtype):
    """Convert TensorRT dtype to numpy dtype."""
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
    }
    return mapping.get(trt_dtype, np.float32)
