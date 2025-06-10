# 使用已配置好的镜像作为基础
FROM sparktts_base_dev:latest

# 设置工作目录
WORKDIR /app

# 复制整个项目到容器
COPY . /app/

# generate_grpc_code.py
RUN python /app/grpc_service/generate_grpc_code.py

# 启动gRPC服务
CMD ["tail", "-f", "/dev/null"] 