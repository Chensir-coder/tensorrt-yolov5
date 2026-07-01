## 转换onnx 
```
cd yolov5 && python export.py   --weights ../weight/yolov5s.pt   --include onnx   --dynamic   --simplify   --opset 11  --batch-size 1
```

## 用 trtexec 构建 FP32 / FP16 引擎

cd /workspace/weight

### FP32
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp32.engine --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --workspace=8192 --verbose

### FP16
CUDA_VISIBLE_DEVICES=1 trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp16.engine --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --workspace=8192 --verbose --fp16