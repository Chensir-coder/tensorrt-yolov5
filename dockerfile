# ============================================================
# 基于 TensorRT 的 YOLOv5s 推理部署镜像
# ============================================================
FROM nvcr.io/nvidia/tensorrt:22.12-py3

WORKDIR /workspace

# 配置 pip 镜像源（加速下载）
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com
ENV PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------
# 层1：安装 PyTorch（体积最大，变更最少，单独放一层）
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ------------------------------------------------------------
# 层2：安装 YOLOv5 基础依赖（相对稳定，单独放一层）
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy>=1.23.5 \
    opencv-python-headless>=4.6.0 \
    pillow>=10.3.0 \
    matplotlib>=3.3 \
    pandas>=1.1.4 \
    scipy>=1.4.1

# ------------------------------------------------------------
# 层3：安装 YOLOv5 轻量/工具依赖（经常变动，单独放一层）cd
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    tqdm>=4.66.3 \
    PyYAML>=5.3.1 \
    requests>=2.32.2 \
    psutil \
    thop>=0.1.1 \
    seaborn>=0.11.0 \
    packaging \
    setuptools>=70.0.0 \
    gitpython>=3.1.30

# ------------------------------------------------------------
# 层4：安装 ultralytics（YOLOv5 核心库，更新频繁，单独放一层）
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    ultralytics>=8.4.65

# ------------------------------------------------------------
# 层5：安装 TensorRT 部署相关依赖（相对稳定）
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    onnx>=1.10.0 \
    onnx-simplifier \
    onnxslim>=0.1.82 \
    pycuda

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxcb-xv0 \
    && rm -rf /var/lib/apt/lists/*