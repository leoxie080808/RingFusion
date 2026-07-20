"""Shared TensorRT load + single-shot inference for the two RingFusion engines
(the distilled backbone and the residual refiner). Factored out of backbone.py
so both nets use one code path.

TensorRT's execution API changed between JetPack releases and this handles both:
  JetPack 5.x -> TRT 8.x : execute_async_v2, index-based bindings
  JetPack 6.x -> TRT 10  : execute_async_v3, name-based tensor addresses

`tensorrt` and `pycuda` only exist on the Jetson, so they are imported lazily
inside __init__ -- importing this module on a dev PC (no CUDA) is fine; only
*instantiating* TRTRunner requires the runtime. Both engines here are single
input / single output with static shapes, which is all this wraps.
"""
import numpy as np


class TRTRunner:
    """One fixed-shape TensorRT engine: feed a contiguous input array, get the
    output array back. Buffers are allocated once and reused every frame."""

    def __init__(self, engine_path, in_shape, out_shape, dtype=np.float32):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)

        self._cuda = cuda
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)
        self.dtype = np.dtype(dtype)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        itemsize = self.dtype.itemsize
        self.d_in = cuda.mem_alloc(int(np.prod(self.in_shape)) * itemsize)
        self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape)) * itemsize)
        self.h_out = cuda.pagelocked_empty(self.out_shape, self.dtype)

        # TRT 10 addresses tensors by name; TRT 8 binds by index order.
        self.trt10 = hasattr(self.ctx, "execute_async_v3")
        if self.trt10:
            names = [self.engine.get_tensor_name(i)
                     for i in range(self.engine.num_io_tensors)]
            self.in_name, self.out_name = names[0], names[1]

    def run(self, x):
        """x: array broadcastable to in_shape, dtype-castable. Returns a COPY of
        the output shaped out_shape (safe to keep; the pinned buffer is reused)."""
        cuda = self._cuda
        x = np.ascontiguousarray(x, dtype=self.dtype)
        if x.shape != self.in_shape:
            raise ValueError(f"input {x.shape} != engine input {self.in_shape}")
        cuda.memcpy_htod_async(self.d_in, x, self.stream)
        if self.trt10:
            self.ctx.set_tensor_address(self.in_name, int(self.d_in))
            self.ctx.set_tensor_address(self.out_name, int(self.d_out))
            self.ctx.execute_async_v3(self.stream.handle)
        else:
            self.ctx.execute_async_v2([int(self.d_in), int(self.d_out)],
                                      self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return np.array(self.h_out)   # copy out of the reused pinned buffer
