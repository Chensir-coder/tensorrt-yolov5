# 1. 选择基础镜像（就像选择地基）
FROM nvcr.io/nvidia/tensorrt:22.12-py3

# 2. 设置工作目录（进入容器后默认的位置）
WORKDIR /workspace

# 3. 安装所有你需要的额外Python包（一次配置，永久享用）
RUN pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    flask 

# docker build -t trt-yolov5:latest .

# 4. 运行容器（启动你的应用环境）
# docker run --gpus all -it --rm -v $(pwd):/workspace trt-yolov5:latest bash
    