# ============================================================
# 基于 TensorRT 的 YOLOv5s 推理部署镜像
# 基础镜像: TensorRT 22.12 (CUDA 11.8 + TensorRT 8.5.x + Python 3.8)
# ============================================================
FROM nvcr.io/nvidia/tensorrt:22.12-py3

# 设置工作目录
WORKDIR /workspace

# 设置 pip 阿里云镜像源（加速国内下载）
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com
ENV PIP_NO_CACHE_DIR=1

# 1. 安装 PyTorch（需要 CUDA 版本，从官方源安装）
RUN pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 2. 安装 YOLOv5 核心依赖
RUN pip install --no-cache-dir \
    gitpython>=3.1.30 \
    matplotlib>=3.3 \
    numpy>=1.23.5 \
    opencv-python>=4.6.0 \
    pillow>=10.3.0 \
    psutil \
    PyYAML>=5.3.1 \
    requests>=2.32.2 \
    scipy>=1.4.1 \
    thop>=0.1.1 \
    tqdm>=4.66.3 \
    pandas>=1.1.4 \
    seaborn>=0.11.0 \
    packaging \
    setuptools>=70.0.0 \
    ultralytics>=8.4.65

# 3. 安装 TensorRT 部署相关依赖（ONNX 导出 + INT8 校准）
RUN pip install --no-cache-dir \
    onnx>=1.10.0 \
    onnx-simplifier \
    onnxslim>=0.1.82 \
    pycuda

# docker build -t trt-yolov5:latest .
# docker run --gpus all -it --rm -v $(pwd):/workspace trt-yolov5:latest bash
