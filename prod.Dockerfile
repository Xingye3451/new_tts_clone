# 使用已配置好的镜像作为基础
FROM sparktts_base:latest

# 设置工作目录
WORKDIR /app

# 复制整个项目到容器
COPY . /app/

# 尝试配置国内源加速（如果失败则使用原有源）
RUN (if [ -f /etc/debian_version ]; then \
    # Debian 系统使用阿里云 Debian 源
    if [ -f /etc/apt/sources.list ]; then \
    cp /etc/apt/sources.list /etc/apt/sources.list.bak; \
    fi && \
    DEBIAN_VERSION=$(cat /etc/debian_version | cut -d. -f1) && \
    if [ "$DEBIAN_VERSION" = "12" ]; then \
    DEBIAN_CODENAME="bookworm"; \
    elif [ "$DEBIAN_VERSION" = "11" ]; then \
    DEBIAN_CODENAME="bullseye"; \
    else \
    DEBIAN_CODENAME="bookworm"; \
    fi && \
    echo "deb https://mirrors.aliyun.com/debian/ ${DEBIAN_CODENAME} main non-free contrib" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ ${DEBIAN_CODENAME}-security main" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ ${DEBIAN_CODENAME}-updates main non-free contrib" >> /etc/apt/sources.list; \
    elif [ -f /etc/lsb-release ]; then \
    # Ubuntu 系统使用阿里云 Ubuntu 源
    if [ -f /etc/apt/sources.list ]; then \
    cp /etc/apt/sources.list /etc/apt/sources.list.bak; \
    fi && \
    UBUNTU_CODENAME=$(lsb_release -cs) && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ ${UBUNTU_CODENAME} main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ ${UBUNTU_CODENAME}-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ ${UBUNTU_CODENAME}-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ ${UBUNTU_CODENAME}-backports main restricted universe multiverse" >> /etc/apt/sources.list; \
    else \
    echo "未检测到支持的系统类型，保持原有源配置"; \
    fi) || (echo "源配置失败，恢复原有配置" && if [ -f /etc/apt/sources.list.bak ]; then mv /etc/apt/sources.list.bak /etc/apt/sources.list; fi)

# 安装字体（包含中文字体支持，清理缓存节省空间）
RUN apt-get update && \
    apt-get install -y \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto-cjk \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    fontconfig && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装依赖
RUN pip install -r /app/grpc_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# generate_grpc_code.py
RUN python /app/grpc_service/generate_grpc_code.py

# 暴露端口
EXPOSE 50051

# 启动生产环境优化的gRPC服务
CMD ["python", "/app/grpc_service/run_service.py", "--workers", "15"] 