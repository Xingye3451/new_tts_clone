#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行gRPC语音服务的生产环境优化脚本
"""

import os
import sys
import argparse
from concurrent import futures
import grpc
import logging

# 确保当前目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入服务相关模块
import voice_service_pb2
import voice_service_pb2_grpc
from main_service import VoiceServiceServicer

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def serve(port=50051, max_workers=5):
    """启动生产环境优化的gRPC服务器"""

    # 生产环境gRPC服务器选项配置
    options = [
        # 连接相关配置
        ("grpc.keepalive_time_ms", 30000),  # 30秒发送keepalive ping
        ("grpc.keepalive_timeout_ms", 5000),  # keepalive ping超时时间5秒
        ("grpc.keepalive_permit_without_calls", True),  # 允许无调用时发送keepalive
        ("grpc.http2.max_pings_without_data", 0),  # 允许无数据时发送ping
        ("grpc.http2.min_time_between_pings_ms", 10000),  # ping间隔最小10秒
        (
            "grpc.http2.min_ping_interval_without_data_ms",
            300000,
        ),  # 无数据时ping间隔5分钟
        # 消息大小限制 (适合大型语音文件传输)
        ("grpc.max_send_message_length", 200 * 1024 * 1024),  # 200MB发送限制
        ("grpc.max_receive_message_length", 200 * 1024 * 1024),  # 200MB接收限制
        # 连接池配置
        ("grpc.max_connection_idle_ms", 300000),  # 连接空闲5分钟后关闭
        ("grpc.max_connection_age_ms", 1800000),  # 连接最大存活30分钟
        ("grpc.max_connection_age_grace_ms", 60000),  # 连接优雅关闭等待1分钟
        # 并发控制
        ("grpc.max_concurrent_streams", max_workers),  # 最大并发流数量
        # 资源限制
        ("grpc.so_reuseport", 1),  # 启用端口复用
    ]

    # 创建线程池执行器，限制最大工作线程数
    thread_pool = futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="grpc-worker"
    )

    # 创建gRPC服务器
    server = grpc.server(thread_pool, options=options)

    # 添加服务
    voice_service_pb2_grpc.add_VoiceServiceServicer_to_server(
        VoiceServiceServicer(), server
    )

    # 绑定端口
    server.add_insecure_port(f"[::]:{port}")

    # 启动服务器
    server.start()

    logger.info(f"生产环境gRPC服务已启动")
    logger.info(f"监听端口: {port}")
    logger.info(f"最大工作线程数: {max_workers}")
    logger.info(f"最大并发任务数: {max_workers}")
    logger.info(f"消息大小限制: 200MB")
    logger.info(f"连接keepalive: 30秒")
    logger.info(f"连接最大存活时间: 30分钟")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在优雅关闭服务...")
        # 优雅关闭，给正在处理的请求60秒完成时间
        server.stop(grace=60)
        logger.info("服务已停止")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动生产环境语音合成gRPC服务")
    parser.add_argument(
        "--port", type=int, default=50051, help="服务监听端口 (默认: 50051)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="最大工作线程数，建议设置为CPU核心数或预期最大并发任务数 (默认: 5)",
    )
    args = parser.parse_args()

    # 验证参数
    if args.workers < 1:
        logger.error("工作线程数必须大于0")
        sys.exit(1)

    if args.workers > 10:
        logger.warning(f"工作线程数 {args.workers} 较大，可能会消耗过多资源")

    serve(port=args.port, max_workers=args.workers)
