#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import logging
import uuid
import numpy as np
from typing import Optional, Dict, Tuple, List, Any, Callable

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("video_processor.log")],
)
logger = logging.getLogger("video_processor")


class VideoProcessor:
    """视频处理器，用于处理视频相关操作"""

    def __init__(self, output_dir: str = "outputs"):
        """
        初始化视频处理器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("视频处理器初始化完成")

    def extract_audio(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        sample_rate: int = 44100,
        mono: bool = True,
        start_time: float = 0.0,
        duration: float = 10.0,
        auto_detect_voice: bool = False,
        max_silence: float = 2.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> str:
        """
        从视频中提取音频

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录，如果不指定则使用默认目录
            sample_rate: 采样率，默认44100
            mono: 是否单声道，默认True
            start_time: 开始时间（秒），默认为0
            duration: 持续时间（秒），默认为10秒，设为0表示提取到结尾
            auto_detect_voice: 是否自动检测有声部分，默认False
            max_silence: 最大静音时长（秒），用于自动检测，默认2.0
            progress_callback: 进度回调函数

        Returns:
            生成的音频文件路径
        """
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            if progress_callback:
                progress_callback(-1, f"错误: 视频文件不存在: {video_path}")
            return ""

        # 使用指定输出目录或默认目录
        output_directory = output_dir if output_dir else self.output_dir
        os.makedirs(output_directory, exist_ok=True)

        # 创建输出文件路径
        timestamp = int(time.time())
        basename = os.path.splitext(os.path.basename(video_path))[0]
        time_info = f"{start_time:.1f}s"
        if duration > 0:
            time_info += f"_{duration:.1f}s"
        else:
            time_info += "_end"

        output_filename = f"audio_{basename}_{time_info}_{timestamp}.wav"
        output_path = os.path.join(output_directory, output_filename)

        try:
            if progress_callback:
                progress_callback(10, "开始提取音频...")

            # 构建FFmpeg命令
            command = ["ffmpeg", "-y"]  # 覆盖已存在文件

            # 添加起始时间参数
            if start_time > 0:
                command.extend(["-ss", str(start_time)])

            # 添加输入文件
            command.extend(["-i", video_path])

            # 添加持续时间参数
            if duration > 0:
                command.extend(["-t", str(duration)])

            # 指定输出格式
            command.extend(
                [
                    "-vn",  # 禁用视频流
                    "-acodec",
                    "pcm_s16le",  # WAV编码
                    "-ar",
                    str(sample_rate),  # 采样率
                    "-ac",
                    "1" if mono else "2",  # 声道数
                    output_path,
                ]
            )

            logger.info(f"执行命令: {' '.join(command)}")

            if progress_callback:
                progress_callback(30, "正在提取音频...")

            # 执行命令
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # 等待进程完成
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                logger.error(f"提取音频失败: {error_msg}")
                if progress_callback:
                    progress_callback(-1, f"提取音频失败: {error_msg[:100]}...")
                return ""

            # 如果需要自动检测有声部分
            if auto_detect_voice and os.path.exists(output_path):
                if progress_callback:
                    progress_callback(70, "正在检测有声部分...")

                try:
                    # 导入项目根目录下的音频工具
                    from audio_utils import trim_voice_section

                    # 进行有声部分检测和切割
                    trimmed_path = trim_voice_section(
                        output_path, max_silence=max_silence, min_voice_duration=1.0
                    )

                    if trimmed_path and os.path.exists(trimmed_path):
                        logger.info(f"检测到有声部分，保存到: {trimmed_path}")
                        output_path = trimmed_path
                except ImportError:
                    logger.warning("未找到音频工具模块，跳过有声部分检测")
                except Exception as e:
                    logger.error(f"检测有声部分时发生错误: {e}")

            if progress_callback:
                progress_callback(100, "音频提取完成")

            logger.info(f"音频提取成功，保存到: {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"提取音频时发生错误: {e}")
            if progress_callback:
                progress_callback(-1, f"错误: {str(e)}")
            return ""


# 直接运行测试
if __name__ == "__main__":
    processor = VideoProcessor()
    # 测试不同参数组合
    print("测试1: 提取前10秒")
    result1 = processor.extract_audio("./inputs/test.mp4", start_time=0, duration=10)
    print(f"提取结果: {result1}")

    print("测试2: 从30秒开始提取5秒")
    result2 = processor.extract_audio("./inputs/test.mp4", start_time=30, duration=5)
    print(f"提取结果: {result2}")
