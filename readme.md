# YOLOv5s TensorRT High-Performance Inference

End-to-end deployment optimization of YOLOv5s on NVIDIA GPU: **PyTorch → ONNX → TensorRT (FP32 / FP16 / INT8)**, delivering ~4× speedup with minimal accuracy loss via custom INT8 calibration.

> 🚧 Work in progress — this project is under active development.

## Overview

- Export YOLOv5s from PyTorch to ONNX with dynamic batch support
- Build TensorRT engines across FP32 / FP16 / INT8 precisions
- Hand-written `IInt8Calibrator` using Entropy (KL-divergence) calibration for high-quality INT8 quantization
- Python inference SDK with async post-processing (CPU-NMS)
- Automated benchmark comparing latency, throughput, and mAP across all precisions


