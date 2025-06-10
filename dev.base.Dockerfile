# 使用已配置好的镜像作为基础
FROM sparktts_base:latest

# 设置工作目录
WORKDIR /app

# 复制整个项目到容器
COPY . /app/



# 安装依赖
RUN pip install -r /app/grpc_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# generate_grpc_code.py
RUN python /app/grpc_service/generate_grpc_code.py

# 启动gRPC服务
CMD ["/bin/bash"] 