#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入语音处理器
from voice_pipeline import VoiceProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("voice_service.log")],
)
logger = logging.getLogger("voice_processor_wrapper")


class VoiceProcessorWithProgress:
    """带进度回调的语音处理器包装器"""

    def __init__(
        self,
        model_dir: str = "pretrained_models/Spark-TTS-0.5B",
        whisper_model_size: str = "medium",
        compute_type: str = "float16",
        language: str = "zh",
        output_dir: str = "outputs",
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        """
        初始化语音处理器包装器

        Args:
            model_dir: Spark-TTS模型路径
            whisper_model_size: Whisper模型大小，可选 tiny/base/small/medium/large-v2
            compute_type: 计算类型，可选 int8/float16/float32
            language: 语言，可选 zh/en
            output_dir: 输出目录
            progress_callback: 进度回调函数，接收进度百分比和消息
        """
        self.output_dir = output_dir
        self.progress_callback = progress_callback

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 延迟初始化VoiceProcessor，直到真正需要时再创建
        self._processor = None
        self._processor_lock = threading.Lock()

        # 保存配置参数
        self.model_dir = model_dir
        self.whisper_model_size = whisper_model_size
        self.compute_type = compute_type
        self.language = language

        logger.info("语音处理器包装器初始化完成")

    def _get_processor(self) -> VoiceProcessor:
        """获取或创建VoiceProcessor实例"""
        if self._processor is None:
            with self._processor_lock:
                if self._processor is None:
                    logger.info("正在初始化VoiceProcessor...")
                    logger.info(f"使用Whisper模型: {self.whisper_model_size}")
                    logger.info(f"使用识别语言: {self.language}")
                    logger.info(f"使用计算精度: {self.compute_type}")

                    if self.progress_callback:
                        self.progress_callback(5, "正在加载语音处理模型...")
                        self.progress_callback(
                            6, f"使用Whisper模型: {self.whisper_model_size}"
                        )
                        self.progress_callback(7, f"使用识别语言: {self.language}")
                        self.progress_callback(8, f"使用计算精度: {self.compute_type}")

                    self._processor = VoiceProcessor(
                        model_dir=self.model_dir,
                        whisper_model_size=self.whisper_model_size,
                        compute_type=self.compute_type,
                        language=self.language,
                        output_dir=self.output_dir,
                    )

                    if self.progress_callback:
                        self.progress_callback(10, "语音处理模型加载完成")

                    logger.info("VoiceProcessor初始化完成")

        return self._processor

    def clone_voice(
        self,
        source_audio: str,
        target_text: str,
        voice_speed: Optional[float] = None,
        prompt_text: Optional[str] = None,
    ) -> Tuple[List[Dict], str]:
        """
        使用源音频的声音特征来合成目标文本的音频

        Args:
            source_audio: 源音频文件路径
            target_text: 目标文本
            voice_speed: 语音速度

        Returns:
            元组，包含源音频识别结果和生成的音频文件路径
        """
        processor = self._get_processor()

        try:
            # 更新进度
            if self.progress_callback:
                self.progress_callback(15, "开始识别源音频...")

            # 检查源音频文件
            if not os.path.exists(source_audio):
                logger.error(f"源音频文件不存在: {source_audio}")
                if self.progress_callback:
                    self.progress_callback(
                        -1, f"错误: 源音频文件不存在: {source_audio}"
                    )
                return [], ""

            # 识别源音频
            logger.info(f"开始识别源音频: {source_audio}")
            recognition_results = processor.transcribe(source_audio)

            # 更新进度
            if self.progress_callback:
                self.progress_callback(30, "源音频识别完成，开始合成新音频...")

            # 如果未提供目标文本，使用识别结果
            if not target_text and recognition_results:
                target_text = recognition_results[0]["text"]
                logger.info(f"未提供目标文本，使用识别结果: {target_text}")

            if not target_text:
                logger.error("未提供目标文本且识别结果为空")
                if self.progress_callback:
                    self.progress_callback(-1, "错误: 未提供目标文本且识别结果为空")
                return recognition_results, ""

            # 合成音频
            logger.info(f"开始合成音频，使用目标文本: {target_text}")

            # 更新进度
            if self.progress_callback:
                self.progress_callback(40, "开始合成音频...")

            # 调用process_voice方法，内部有详细的进度控制
            recognition_results, output_path = processor.process_voice(
                source_audio=source_audio,
                target_text=target_text,
                voice_speed=voice_speed,
                prompt_text=prompt_text,
            )

            if output_path:
                logger.info(f"音频合成完成，保存到: {output_path}")
                if self.progress_callback:
                    self.progress_callback(100, f"合成完成，输出文件: {output_path}")
                return recognition_results, output_path
            else:
                logger.error("音频合成失败")
                if self.progress_callback:
                    self.progress_callback(-1, "音频合成失败")
                return recognition_results, ""

        except Exception as e:
            logger.exception(f"克隆音色过程发生错误: {e}")
            if self.progress_callback:
                self.progress_callback(-1, f"错误: {str(e)}")
            return [], ""

    def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        voice_speed: Optional[float] = None,
        prompt_text: Optional[str] = "",
    ) -> str:
        """
        使用预设声音合成文本

        Args:
            text: 要合成的文本
            voice_name: 声音名称，默认为None使用默认声音
            voice_speed: 语速，默认为None使用正常语速

        Returns:
            生成的音频文件路径
        """
        processor = self._get_processor()

        try:
            # 更新进度
            if self.progress_callback:
                self.progress_callback(10, "准备合成音频...")
                self.progress_callback(
                    11, f"使用Whisper模型: {self.whisper_model_size}"
                )

            # 记录使用的模型信息
            logger.info(f"合成音频使用Whisper模型: {self.whisper_model_size}")
            logger.info(f"合成音频使用识别语言: {self.language}")
            logger.info(f"合成音频使用计算精度: {self.compute_type}")

            # 检查输入文本
            if not text:
                logger.error("未提供文本内容")
                if self.progress_callback:
                    self.progress_callback(-1, "错误: 未提供文本内容")
                return ""

            # 更新进度
            if self.progress_callback:
                self.progress_callback(30, "开始合成音频...")

            # 合成音频
            output_path = processor.synthesize(
                text=text,
                voice_name=voice_name,
                voice_speed=voice_speed,
                prompt_text=prompt_text,
            )

            if output_path:
                logger.info(f"音频合成完成，保存到: {output_path}")
                if self.progress_callback:
                    self.progress_callback(100, f"合成完成，输出文件: {output_path}")
                return output_path
            else:
                logger.error("音频合成失败")
                if self.progress_callback:
                    self.progress_callback(-1, "音频合成失败")
                return ""

        except Exception as e:
            logger.exception(f"合成音频过程发生错误: {e}")
            if self.progress_callback:
                self.progress_callback(-1, f"错误: {str(e)}")
            return ""

    def transcribe(self, audio_path: str):
        """
        使用Whisper识别音频内容

        Args:
            audio_path: 音频文件路径

        Returns:
            识别结果列表
        """
        processor = self._get_processor()

        try:
            if self.progress_callback:
                self.progress_callback(20, "开始识别音频...")

            # 调用底层VoiceProcessor的transcribe方法
            results = processor.transcribe(audio_path)

            if self.progress_callback:
                self.progress_callback(30, "音频识别完成")

            return results
        except Exception as e:
            logger.exception(f"识别音频时发生错误: {e}")
            if self.progress_callback:
                self.progress_callback(-1, f"识别音频失败: {str(e)}")
            return []
