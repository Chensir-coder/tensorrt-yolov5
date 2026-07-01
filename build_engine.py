"""
TensorRT Engine Builder for YOLOv5s

Usage:
    # FP32
    python build_engine.py --onnx weight/yolov5s.onnx --output weight/build_engine_yolov5s_fp32.engine

    # FP16
    python build_engine.py --onnx weight/yolov5s.onnx --output weight/build_engine_yolov5s_fp16.engine --fp16

    # INT8 (need calibrator, see Phase 3)
    python build_engine.py --onnx weight/yolov5s.onnx --output weight/build_engine_yolov5s_int8.engine --int8
"""

import argparse
import sys
import os

import tensorrt as trt


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = False,
    int8: bool = False,
    workspace: int = 8192,         # MB
    min_shapes: tuple = (1, 3, 640, 640),
    opt_shapes: tuple = (1, 3, 640, 640),
    max_shapes: tuple = (1, 3, 640, 640),
    calibrator: trt.IInt8Calibrator = None,
):
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path:    Path to the input ONNX file.
        engine_path:  Path to save the serialized engine.
        fp16:         Enable FP16 precision.
        int8:         Enable INT8 precision (requires calibrator).
        workspace:    Max workspace size in MB.
        min_shapes:   Min input shape (batch, channel, height, width).
        opt_shapes:   Optimal input shape.
        max_shapes:   Max input shape.
        calibrator:   IInt8Calibrator instance (required for INT8).
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # 1. Build network from ONNX
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[ERROR] {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model.")

    print(f"[INFO] ONNX loaded: {onnx_path}")

    # 2. Create build config
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * (1 << 20)  # MB -> bytes

    # Precision flags
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 mode enabled.")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is not None:
            config.int8_calibrator = calibrator
        print("[INFO] INT8 mode enabled.")

    # 3. Set dynamic shape profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, min_shapes, opt_shapes, max_shapes)
    config.add_optimization_profile(profile)

    print(f"[INFO] Input  '{input_name}': min={min_shapes}, opt={opt_shapes}, max={max_shapes}")
    print(f"[INFO] Workspace: {workspace} MB")
    print("[INFO] Building engine...")

    # 4. Build serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed.")

    # 5. Save to disk
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[INFO] Engine saved: {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--output", type=str, required=True, help="Path to output engine file")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision")
    parser.add_argument("--workspace", type=int, default=8192, help="Max workspace in MB (default: 8192)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--height", type=int, default=640, help="Input height (default: 640)")
    parser.add_argument("--width", type=int, default=640, help="Input width (default: 640)")
    args = parser.parse_args()

    batch, c, h, w = args.batch, 3, args.height, args.width
    shape = (batch, c, h, w)

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        fp16=args.fp16,
        int8=args.int8,
        workspace=args.workspace,
        min_shapes=shape,
        opt_shapes=shape,
        max_shapes=shape,
    )


if __name__ == "__main__":
    main()
