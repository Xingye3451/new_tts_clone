#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成gRPC服务代码的脚本
"""

import os
import sys
import subprocess
from pathlib import Path


def generate_grpc_code():
    """生成gRPC代码"""
    # 获取当前脚本目录
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # proto文件路径
    proto_file = current_dir / "voice_service.proto"

    # 检查proto文件是否存在
    if not proto_file.exists():
        print(f"错误：proto文件不存在: {proto_file}")
        return False

    print(f"使用proto文件: {proto_file}")

    # 生成Python代码的命令
    command = [
        "python",
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={current_dir}",
        f"--python_out={current_dir}",
        f"--grpc_python_out={current_dir}",
        str(proto_file),
    ]

    try:
        print(f"执行命令: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True)
        print("gRPC代码生成成功!")

        # 打印生成的文件
        generated_files = [
            current_dir / f"{proto_file.stem}_pb2.py",
            current_dir / f"{proto_file.stem}_pb2_grpc.py",
        ]

        for file in generated_files:
            if file.exists():
                print(f"生成文件: {file}")
            else:
                print(f"警告: 文件未生成: {file}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: gRPC代码生成失败: {e}")
        print(f"错误输出: {e.stderr.decode('utf-8', errors='replace')}")
        return False


if __name__ == "__main__":
    # 检查grpcio-tools是否已安装
    try:
        import grpc_tools.protoc

        print("已安装grpcio-tools")
    except ImportError:
        print("错误: 未安装grpcio-tools，请先使用pip安装：")
        print("pip install grpcio-tools")
        sys.exit(1)

    # 生成代码
    if generate_grpc_code():
        print("代码生成完成")
        sys.exit(0)
    else:
        print("代码生成失败")
        sys.exit(1)
