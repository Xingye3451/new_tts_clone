# 使用已构建的基础镜像
FROM breakstring/spark-tts:latest-full

# 设置工作目录
WORKDIR /app

# 删除/app下所有文件
RUN rm -rf /app/*

# 安装nc
RUN  apt-get update && apt-get install -y netcat-openbsd && apt-get clean

# 设置默认命令
CMD ["/bin/bash"]