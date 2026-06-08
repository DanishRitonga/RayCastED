"""RayCastED — Jetson Inference Runtime.

Loads a TensorRT engine, runs inference on WSI tiles, and post-processes
raw head output into polygon detections. Runs on NVIDIA Jetson (Orin/Xavier).

Usage:
    runtime = JetsonRuntime('polygon_yolo.engine', 'polygon_yolo.meta.json')
    polygons = runtime.infer(tile_image)
"""

import json

import numpy as np

from raycasted.export.postprocess import postprocess_raw_output


class JetsonRuntime:
    """TensorRT inference runtime for RayCastED on Jetson."""

    def __init__(self, engine_path: str, meta_path: str):
        self.meta = self._load_meta(meta_path)
        self.imgsz = self.meta['imgsz']
        self.nc = self.meta.get('nc', 5)
        self.n_rays = self.meta.get('n_rays', 64)
        self.conf_threshold = self.meta.get('conf_threshold', 0.20)
        self.binary_threshold = self.meta.get('binary_threshold', 0.01)
        self.strides = self.meta.get('strides', [4, 8, 16])
        self.hierarchical = self.meta.get('hierarchical_cls', False)
        self.raycast_dim = 2 + self.n_rays

        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self._setup_buffers()

    @staticmethod
    def _load_meta(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def _load_engine(self, path: str):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(path, 'rb') as f:
            return runtime.deserialize_cuda_engine(f.read())

    def _setup_buffers(self):
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self.bindings = []
        self.stream = cuda.Stream()
        self._input = {}
        self._output = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = _trt_to_np(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorMode.INPUT:
                self._input = {'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape}
            else:
                self._output[name] = {'host': host_mem, 'device': device_mem, 'shape': shape}

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2
        resized = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return np.ascontiguousarray(blob)

    def infer(self, image: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda

        blob = self.preprocess(image)
        np.copyto(self._input['host'], blob.ravel())

        cuda.memcpy_htod_async(self._input['device'], self._input['host'], self.stream)
        self.context.execute_async_v3(self.stream.handle)

        for name, buf in self._output.items():
            cuda.memcpy_dtoh_async(buf['host'], buf['device'], self.stream)
        self.stream.synchronize()

        boxes_raw = self._output['boxes']['host'].reshape(-1, self.raycast_dim)

        if self.hierarchical:
            binary_raw = self._output['binary']['host'].reshape(-1)
            class_raw = self._output['class']['host'].reshape(-1, self.nc)
            return postprocess_raw_output(
                boxes_raw, binary_raw, class_raw,
                strides=self.strides, imgsz=self.imgsz,
                conf_threshold=self.conf_threshold,
                binary_threshold=self.binary_threshold,
                n_rays=self.n_rays,
            )
        else:
            scores_raw = self._output['scores']['host'].reshape(-1, self.nc)
            return postprocess_raw_output(
                boxes_raw, None, scores_raw,
                strides=self.strides, imgsz=self.imgsz,
                conf_threshold=self.conf_threshold,
                n_rays=self.n_rays,
            )


def _trt_to_np(trt_dtype):
    import tensorrt as trt
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
    }
    return mapping.get(trt_dtype, np.float32)
