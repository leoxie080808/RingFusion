"""Build a TensorRT engine from ONNX. RUN THIS ON THE JETSON -- engines are
hardware- and TensorRT-version-specific and are not portable.

  Backbone: build INT8 with a calibration set of 200-500 REAL deployment images
            (do not calibrate on synthetic data -- activation ranges differ, §7.2).
  Residual: build FP16, not INT8 -- it is small and quantizing a variance head
            degrades calibration (§7.2).

Usage (on the Orin):
    python tools/build_engine.py --onnx student.onnx --out student_int8.engine \
        --precision int8 --calib-dir calib_images/ --calib-cache student.cache
    python tools/build_engine.py --onnx residual.onnx --out residual_fp16.engine \
        --precision fp16
"""
import argparse
import glob
import os

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _preprocess_image(path, hw):
    """Match TensorRTBackbone._preprocess: resize, /255, ImageNet-normalize, CHW."""
    import cv2
    h, w = hw
    img = cv2.imread(path)[:, :, ::-1]                       # BGR->RGB
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    img = (img - mean) / std
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds preprocessed real images to the INT8 calibrator (backbone, 3-channel)."""

    def __init__(self, calib_dir, cache_path, input_hw, batch=1):
        super().__init__()
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self.cuda = cuda
        self.cache_path = cache_path
        self.hw = input_hw
        self.batch = batch
        self.files = sorted(sum([glob.glob(os.path.join(calib_dir, e))
                                 for e in ('*.jpg', '*.jpeg', '*.png')], []))
        if not self.files:
            raise SystemExit(f"no calibration images in {calib_dir}")
        self.idx = 0
        self.device_input = cuda.mem_alloc(
            int(np.prod((batch, 3, *input_hw))) * 4)

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        if self.idx + self.batch > len(self.files):
            return None
        imgs = [_preprocess_image(self.files[self.idx + i], self.hw)
                for i in range(self.batch)]
        arr = np.ascontiguousarray(np.concatenate(imgs, axis=0), dtype=np.float32)
        self.cuda.memcpy_htod(self.device_input, arr)
        self.idx += self.batch
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, 'wb') as f:
            f.write(cache)


def build(args):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(args.onnx, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 30)

    if args.precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.precision == 'int8':
        if not args.calib_dir:
            raise SystemExit("--calib-dir is required for int8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = ImageCalibrator(
            args.calib_dir, args.calib_cache, tuple(args.input_hw))

    print(f"building {args.precision} engine from {args.onnx} ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    with open(args.out, 'wb') as f:
        f.write(serialized)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16')
    ap.add_argument('--calib-dir', default='')
    ap.add_argument('--calib-cache', default='calib.cache')
    ap.add_argument('--input-hw', type=int, nargs=2, default=[288, 384])
    ap.add_argument('--workspace', type=int, default=4, help='GiB')
    build(ap.parse_args())


if __name__ == '__main__':
    main()
