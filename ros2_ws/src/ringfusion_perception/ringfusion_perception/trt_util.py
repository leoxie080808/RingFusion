"""Shared TensorRT load + single-shot inference for the two RingFusion engines
(the distilled backbone and the residual refiner). Factored out of backbone.py
so both nets use one code path.

TensorRT's execution API changed between JetPack releases and this handles both:
  JetPack 5.x -> TRT 8.x : execute_async_v2, index-based bindings
  JetPack 6.x -> TRT 10  : execute_async_v3, name-based tensor addresses

Device memory + the CUDA stream come from **torch**, not pycuda: torch is the CUDA
runtime already installed and working on the Jetson, while pycuda is not present
(and painful to build there). A torch CUDA tensor's `.data_ptr()` is the device
address TensorRT binds to, and `torch.cuda.Stream().cuda_stream` is the stream
handle it enqueues on. `tensorrt`/`torch` are imported lazily inside __init__, so
importing this module on a dev PC (no CUDA) is fine; only *instantiating* TRTRunner
needs the runtime. Both engines here are single input / single output, static shape.
"""
import numpy as np


class TRTRunner:
    """One fixed-shape TensorRT engine: feed a contiguous input array, get the
    output array back. Buffers are allocated once and reused every frame."""

    def __init__(self, engine_path, in_shape, out_shape, dtype=np.float32):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)
        self.dtype = np.dtype(dtype)
        # matching torch dtype for the device buffers
        self._tdtype = torch.from_numpy(np.zeros(1, self.dtype)).dtype

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # persistent device buffers; torch owns the memory, data_ptr() feeds TRT
        self.d_in = torch.empty(self.in_shape, dtype=self._tdtype, device="cuda")
        self.d_out = torch.empty(self.out_shape, dtype=self._tdtype, device="cuda")

        # TRT 10 addresses tensors by name; TRT 8 binds by index order.
        self.trt10 = hasattr(self.ctx, "execute_async_v3")
        if self.trt10:
            names = [self.engine.get_tensor_name(i)
                     for i in range(self.engine.num_io_tensors)]
            self.in_name, self.out_name = names[0], names[1]
            # addresses are stable (buffers persist), so bind once
            self.ctx.set_tensor_address(self.in_name, int(self.d_in.data_ptr()))
            self.ctx.set_tensor_address(self.out_name, int(self.d_out.data_ptr()))

    def run(self, x):
        """x: array broadcastable to in_shape, dtype-castable. Returns a COPY of
        the output shaped out_shape (safe to keep; the device buffer is reused)."""
        torch = self.torch
        x = np.ascontiguousarray(x, dtype=self.dtype)
        if x.shape != self.in_shape:
            raise ValueError(f"input {x.shape} != engine input {self.in_shape}")
        with torch.cuda.stream(self.stream):
            self.d_in.copy_(torch.from_numpy(x), non_blocking=True)
            if self.trt10:
                self.ctx.execute_async_v3(self.stream.cuda_stream)
            else:
                self.ctx.execute_async_v2(
                    [int(self.d_in.data_ptr()), int(self.d_out.data_ptr())],
                    self.stream.cuda_stream)
            host = self.d_out.to("cpu", non_blocking=True)
        self.stream.synchronize()
        return np.ascontiguousarray(host.numpy(), dtype=self.dtype)
