#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import uuid
import threading
import grpc
from concurrent import futures
from typing import Dict, List, Optional
import logging
import torch
import gc

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入proto生成的模块
import voice_service_pb2
import voice_service_pb2_grpc

# 导入进度管理器和语音处理器包装器
from progress_manager import ProgressManager
from voice_processor_wrapper import VoiceProcessorWithProgress
from video_processor import VideoProcessor
from video_subtitle_processor import VideoSubtitleProcessor

# 导入音频处理工具
from audio_utils import check_audio_silence, remove_silence_from_ends

# 获取环境变量
FILES_DIR = os.environ.get("FILES_DIR", "files")
MODEL_DIR = os.environ.get("MODEL_DIR", "pretrained_models/Spark-TTS-0.5B")
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "medium")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
LANGUAGE = os.environ.get("LANGUAGE", "zh")

# 确保文件目录存在
os.makedirs(FILES_DIR, exist_ok=True)

# 创建进度管理器
progress_manager = ProgressManager()

# 创建日志记录器
logger = logging.getLogger(__name__)


def clean_gpu_memory():
    """清理GPU内存"""
    try:
        if torch.cuda.is_available():
            # 清空GPU缓存
            torch.cuda.empty_cache()
            # 强制垃圾回收
            gc.collect()
            logger.info("GPU内存已清理")
    except Exception as e:
        logger.error(f"清理GPU内存时发生错误: {e}")


class VoiceServiceServicer(voice_service_pb2_grpc.VoiceServiceServicer):
    """gRPC声音合成服务实现类"""

    def __init__(self):
        """初始化服务"""
        self.tasks = {}  # 存储任务信息

    def _get_task_directory(self, task_id: str) -> str:
        """获取任务专属目录路径"""
        task_dir = os.path.join(FILES_DIR, f"dir_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        return task_dir

    def CloneVoice(self, request, context):
        """同步音色克隆接口"""
        try:
            # 获取或生成任务ID
            task_id = request.task_id if request.task_id else str(uuid.uuid4())
            task_step = request.task_step if request.task_step else "0"
            prompt_text = request.prompt_text if request.prompt_text else ""
            logger.info(
                f"收到克隆请求 - task_id: {task_id}, prompt_text长度: {len(prompt_text)}"
            )
            # 获取任务专属目录
            task_dir = self._get_task_directory(task_id)

            # 获取请求参数
            audio_name = request.audio_name
            source_audio = os.path.join(FILES_DIR, audio_name)  # 从统一文件目录获取输入
            target_text = request.target_text
            voice_speed = request.voice_speed if request.voice_speed else 1.0

            # ========== 增强输入验证 ==========

            # 验证目标文本
            if not target_text or len(target_text.strip()) == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("目标文本不能为空")
                return voice_service_pb2.CloneVoiceResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message="目标文本不能为空",
                    is_finished=True,
                )

            # 验证文本长度合理性（避免过短或过长）
            target_text = target_text.strip()
            if len(target_text) < 2:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("目标文本过短，至少需要2个字符")
                return voice_service_pb2.CloneVoiceResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message="目标文本过短，至少需要2个字符",
                    is_finished=True,
                )

            if len(target_text) > 1000:
                logger.warning(
                    f"目标文本较长({len(target_text)}字符)，可能影响生成质量"
                )

            # 检查源音频文件是否存在
            if not os.path.exists(source_audio):
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"源音频文件不存在: {source_audio}")
                return voice_service_pb2.CloneVoiceResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message=f"源音频文件不存在: {source_audio}",
                    is_finished=True,
                )

            # 验证音频文件大小和格式
            try:
                audio_size = os.path.getsize(source_audio)
                if audio_size == 0:
                    raise ValueError("音频文件为空")
                if audio_size < 1024:  # 小于1KB的音频文件可能无效
                    logger.warning(f"音频文件过小({audio_size}字节)，可能导致处理失败")
                logger.info(f"源音频文件大小: {audio_size / 1024:.2f} KB")
            except Exception as e:
                logger.error(f"验证音频文件失败: {e}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"音频文件无效: {e}")
                return voice_service_pb2.CloneVoiceResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message=f"音频文件无效: {e}",
                    is_finished=True,
                )

            # 创建语音处理器
            processor = VoiceProcessorWithProgress(
                model_dir=MODEL_DIR,
                whisper_model_size=WHISPER_SIZE,
                compute_type=COMPUTE_TYPE,
                language=LANGUAGE,
                output_dir=task_dir,
            )

            try:
                # 直接执行音色克隆
                logger.info(
                    f"开始克隆音色 - 源音频: {source_audio}, 目标文本: {target_text[:50]}..."
                )
                recognition_results, output_path = processor.clone_voice(
                    source_audio=source_audio,
                    target_text=target_text,
                    voice_speed=voice_speed,
                    prompt_text=prompt_text,
                )

                if output_path:
                    # 验证输出文件是否有效
                    if not os.path.exists(output_path):
                        logger.error("生成的音频文件不存在")
                        return voice_service_pb2.CloneVoiceResponse(
                            task_id=task_id,
                            status=voice_service_pb2.TaskStatus.FAILED,
                            message="生成的音频文件不存在",
                            is_finished=True,
                        )

                    output_size = os.path.getsize(output_path)
                    if output_size == 0:
                        logger.error("生成的音频文件为空")
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        return voice_service_pb2.CloneVoiceResponse(
                            task_id=task_id,
                            status=voice_service_pb2.TaskStatus.FAILED,
                            message="生成的音频文件为空",
                            is_finished=True,
                        )

                    logger.info(f"生成音频文件大小: {output_size / 1024:.2f} KB")

                    # 检查生成的音频是否全部为静音
                    logger.info("开始检测生成音频的静音情况...")
                    is_silent, voice_ratio = check_audio_silence(
                        output_path, threshold=0.005, min_voice_ratio=0.02
                    )

                    # 先尝试移除前后的静音片段
                    logger.info("移除音频前后的静音片段...")
                    processed_path = remove_silence_from_ends(
                        output_path, threshold=0.005
                    )

                    # 如果处理后的音频不同，使用处理后的音频并重新检测
                    if processed_path != output_path:
                        logger.info("音频静音处理完成，使用处理后的音频")
                        output_path = processed_path
                        # 重新检测处理后的音频
                        is_silent, voice_ratio = check_audio_silence(
                            output_path, threshold=0.005, min_voice_ratio=0.02
                        )
                        logger.info(
                            f"处理后音频检测结果: 有声比例={voice_ratio:.2%}, 是否静音={is_silent}"
                        )

                    # 如果处理后仍然是静音或有声部分过少，才返回失败
                    if is_silent:
                        logger.error(
                            f"生成的音频处理后仍然全部为静音或有声部分过少 (有声比例: {voice_ratio:.2%})"
                        )
                        # 删除无效的音频文件
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        return voice_service_pb2.CloneVoiceResponse(
                            task_id=task_id,
                            status=voice_service_pb2.TaskStatus.FAILED,
                            message=f"语音合成失败: 生成的音频全部为静音 (有声比例: {voice_ratio:.2%})",
                            is_finished=True,
                        )

                    # 生成包含任务步骤的输出文件名
                    file_base, file_ext = os.path.splitext(
                        os.path.basename(output_path)
                    )
                    output_filename = f"{file_base}_step{task_step}{file_ext}"

                    # 移动文件到正确的名称
                    new_output_path = os.path.join(task_dir, output_filename)
                    if output_path != new_output_path:
                        os.rename(output_path, new_output_path)

                    # 创建响应
                    response = voice_service_pb2.CloneVoiceResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.COMPLETED,
                        message=f"克隆完成 (有声比例: {voice_ratio:.2%})",
                        is_finished=True,
                        output_filename=output_filename,
                    )

                    # 添加识别结果
                    if recognition_results:
                        for result in recognition_results:
                            segment = voice_service_pb2.TranscriptionSegment(
                                text=result["text"],
                                start=result["start"],
                                end=result["end"],
                            )
                            response.segments.append(segment)

                    return response
                else:
                    return voice_service_pb2.CloneVoiceResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.FAILED,
                        message="克隆失败",
                        is_finished=True,
                    )

            except Exception as e:
                error_message = f"克隆过程发生错误: {str(e)}"
                logger.error(error_message)

                # 记录详细错误信息以便调试
                if "Calculated padded input size" in str(e):
                    logger.error("检测到卷积层输入尺寸错误，可能原因:")
                    logger.error("1. 输入音频数据为空或长度为0")
                    logger.error("2. 目标文本过短导致模型无法生成有效特征")
                    logger.error("3. 音频预处理失败")
                    logger.error("4. GPU内存不足或模型状态异常")

                return voice_service_pb2.CloneVoiceResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message=error_message,
                    is_finished=True,
                )
            finally:
                # 清理GPU内存
                clean_gpu_memory()

        except Exception as e:
            error_message = f"处理请求时发生错误: {str(e)}"
            logger.error(error_message)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_message)
            return voice_service_pb2.CloneVoiceResponse(
                task_id=task_id if "task_id" in locals() else str(uuid.uuid4()),
                status=voice_service_pb2.TaskStatus.FAILED,
                message=error_message,
                is_finished=True,
            )

    def Synthesize(self, request, context):
        """同步音频合成接口"""
        try:
            # 获取或生成任务ID
            task_id = request.task_id if request.task_id else str(uuid.uuid4())
            task_step = request.task_step if request.task_step else "0"
            prompt_text = request.prompt_text if request.prompt_text else ""
            logger.info(
                f"收到合成请求 - task_id: {task_id}, prompt_text长度: {len(prompt_text)}"
            )

            # 获取任务专属目录
            task_dir = self._get_task_directory(task_id)

            # 获取请求参数
            text = request.text
            voice_name = (
                request.voice_name
                if hasattr(request, "voice_name") and request.voice_name
                else None
            )
            voice_speed = request.voice_speed if request.voice_speed else 1.0

            # ========== 增强输入验证 ==========

            # 验证合成文本
            if not text or len(text.strip()) == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("文本内容不能为空")
                return voice_service_pb2.SynthesizeResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message="文本内容不能为空",
                    is_finished=True,
                )

            # 验证文本长度合理性
            text = text.strip()
            if len(text) < 2:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("合成文本过短，至少需要2个字符")
                return voice_service_pb2.SynthesizeResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message="合成文本过短，至少需要2个字符",
                    is_finished=True,
                )

            if len(text) > 1000:
                logger.warning(f"合成文本较长({len(text)}字符)，可能影响生成质量")

            # 获取whisper模型参数
            whisper_model = (
                request.whisper_model
                if hasattr(request, "whisper_model") and request.whisper_model
                else WHISPER_SIZE
            )
            logger.info(f"使用Whisper模型: {whisper_model}")

            # 获取语言参数
            whisper_language = (
                request.whisper_language
                if hasattr(request, "whisper_language") and request.whisper_language
                else LANGUAGE
            )
            logger.info(f"使用识别语言: {whisper_language}")

            # 获取计算精度参数
            compute_type = (
                request.compute_type
                if hasattr(request, "compute_type") and request.compute_type
                else COMPUTE_TYPE
            )
            logger.info(f"使用计算精度: {compute_type}")

            # 如果voice_name是文件名，构建完整路径并验证
            if voice_name:
                original_voice_name = voice_name
                if not os.path.isabs(voice_name) and not os.path.exists(voice_name):
                    potential_voice_path = os.path.join(FILES_DIR, voice_name)
                    if os.path.exists(potential_voice_path):
                        logger.info(
                            f"找到音频文件: {potential_voice_path}，将使用此文件作为声音参考"
                        )
                        voice_name = potential_voice_path
                    else:
                        # 尝试在task_dir中查找
                        task_dir_voice_path = os.path.join(task_dir, voice_name)
                        if os.path.exists(task_dir_voice_path):
                            logger.info(
                                f"在任务目录中找到音频文件: {task_dir_voice_path}，将使用此文件作为声音参考"
                            )
                            voice_name = task_dir_voice_path
                        else:
                            logger.warning(
                                f"无法找到音频文件: {voice_name}，在FILES_DIR或task_dir中均不存在"
                            )
                            voice_name = None  # 设为None以使用默认参数

                # 如果找到了音频文件，验证其有效性
                if voice_name and os.path.exists(voice_name):
                    try:
                        audio_size = os.path.getsize(voice_name)
                        if audio_size == 0:
                            logger.warning(
                                f"参考音频文件为空: {voice_name}，将使用默认参数"
                            )
                            voice_name = None
                        elif audio_size < 1024:
                            logger.warning(
                                f"参考音频文件过小({audio_size}字节): {voice_name}，可能导致处理失败"
                            )
                        else:
                            logger.info(f"参考音频文件大小: {audio_size / 1024:.2f} KB")
                    except Exception as e:
                        logger.error(f"验证参考音频文件失败: {e}，将使用默认参数")
                        voice_name = None

            # 检查文本是否为空（已在上面验证过，这里保留原有逻辑）
            if not text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("文本内容不能为空")
                return voice_service_pb2.SynthesizeResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message="文本内容不能为空",
                    is_finished=True,
                )

            # 创建语音处理器
            processor = VoiceProcessorWithProgress(
                model_dir=MODEL_DIR,
                whisper_model_size=whisper_model,
                compute_type=compute_type,
                language=whisper_language,
                output_dir=task_dir,
            )

            try:
                # 直接执行音频合成
                logger.info(
                    f"开始合成音频 - 文本: {text[:50]}..., 参考音频: {voice_name}"
                )
                output_path = processor.synthesize(
                    text=text,
                    voice_name=voice_name,
                    voice_speed=voice_speed,
                    prompt_text=prompt_text,
                )

                if output_path:
                    # 验证输出文件是否有效
                    if not os.path.exists(output_path):
                        logger.error("生成的音频文件不存在")
                        return voice_service_pb2.SynthesizeResponse(
                            task_id=task_id,
                            status=voice_service_pb2.TaskStatus.FAILED,
                            message="生成的音频文件不存在",
                            is_finished=True,
                        )

                    output_size = os.path.getsize(output_path)
                    if output_size == 0:
                        logger.error("生成的音频文件为空")
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        return voice_service_pb2.SynthesizeResponse(
                            task_id=task_id,
                            status=voice_service_pb2.TaskStatus.FAILED,
                            message="生成的音频文件为空",
                            is_finished=True,
                        )

                    logger.info(f"生成音频文件大小: {output_size / 1024:.2f} KB")

                    # 检查生成的音频是否全部为静音
                    logger.info("开始检测生成音频的静音情况...")
                    is_silent, voice_ratio = check_audio_silence(
                        output_path, threshold=0.005, min_voice_ratio=0.02
                    )

                    # 先尝试移除前后的静音片段
                    logger.info("移除音频前后的静音片段...")
                    processed_path = remove_silence_from_ends(
                        output_path, threshold=0.005
                    )

                    # 如果处理后的音频不同，使用处理后的音频并重新检测
                    if processed_path != output_path:
                        logger.info("音频静音处理完成，使用处理后的音频")
                        output_path = processed_path
                        # 重新检测处理后的音频
                        is_silent, voice_ratio = check_audio_silence(
                            output_path, threshold=0.005, min_voice_ratio=0.02
                        )
                        logger.info(
                            f"处理后音频检测结果: 有声比例={voice_ratio:.2%}, 是否静音={is_silent}"
                        )

                    # 如果处理后仍然是静音或有声部分过少，才返回失败
                    if is_silent:
                        logger.error(
                            f"生成的音频处理后仍然全部为静音或有声部分过少 (有声比例: {voice_ratio:.2%})"
                        )
                        # 删除无效的音频文件
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        return voice_service_pb2.SynthesizeResponse(
                            task_id=task_id,
                            status=voice_service_pb2.TaskStatus.FAILED,
                            message=f"语音合成失败: 生成的音频全部为静音 (有声比例: {voice_ratio:.2%})",
                            is_finished=True,
                        )

                    # 生成包含任务步骤的输出文件名
                    file_base, file_ext = os.path.splitext(
                        os.path.basename(output_path)
                    )
                    output_filename = f"{file_base}_step{task_step}{file_ext}"

                    # 移动文件到正确的名称
                    new_output_path = os.path.join(task_dir, output_filename)
                    if output_path != new_output_path:
                        os.rename(output_path, new_output_path)

                    return voice_service_pb2.SynthesizeResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.COMPLETED,
                        message=f"合成完成 (有声比例: {voice_ratio:.2%})",
                        is_finished=True,
                        output_filename=output_filename,
                    )
                else:
                    return voice_service_pb2.SynthesizeResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.FAILED,
                        message="合成失败",
                        is_finished=True,
                    )

            except Exception as e:
                error_message = f"合成过程发生错误: {str(e)}"
                logger.error(error_message)

                # 记录详细错误信息以便调试
                if "Calculated padded input size" in str(e):
                    logger.error("检测到卷积层输入尺寸错误，可能原因:")
                    logger.error("1. 输入文本过短导致特征张量为空")
                    logger.error("2. 参考音频数据预处理失败")
                    logger.error("3. 模型内部状态异常或GPU内存不足")
                    logger.error("4. 文本编码或音频特征提取失败")

                return voice_service_pb2.SynthesizeResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message=error_message,
                    is_finished=True,
                )
            finally:
                # 清理GPU内存
                clean_gpu_memory()

        except Exception as e:
            error_message = f"处理请求时发生错误: {str(e)}"
            logger.error(error_message)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_message)
            return voice_service_pb2.SynthesizeResponse(
                task_id=task_id if "task_id" in locals() else str(uuid.uuid4()),
                status=voice_service_pb2.TaskStatus.FAILED,
                message=error_message,
                is_finished=True,
            )

    def ExtractAudioFromVideo(self, request, context):
        """从视频中提取音频的接口 - 同步版本"""
        try:
            # 获取请求参数
            video_name = request.video_name

            # 获取或生成任务ID
            task_id = request.task_id if request.task_id else str(uuid.uuid4())
            task_step = request.task_step if request.task_step else "0"

            # 获取任务专属目录
            task_dir = self._get_task_directory(task_id)

            # 设置视频文件查找路径，首先尝试在任务目录中查找，如果不存在则在FILES_DIR中查找
            video_path_in_task_dir = os.path.join(task_dir, video_name)
            video_path_in_files_dir = os.path.join(FILES_DIR, video_name)

            if os.path.exists(video_path_in_task_dir):
                video_path = video_path_in_task_dir
                print(f"在任务目录中找到视频文件: {video_path}")
            elif os.path.exists(video_path_in_files_dir):
                video_path = video_path_in_files_dir
                print(f"在FILES_DIR中找到视频文件: {video_path}")
            else:
                # 如果都找不到，尝试在task_dir的父目录中查找
                parent_dir = os.path.dirname(task_dir)
                video_path_in_parent = os.path.join(parent_dir, video_name)
                if os.path.exists(video_path_in_parent):
                    video_path = video_path_in_parent
                    print(f"在父目录中找到视频文件: {video_path}")
                else:
                    error_msg = f"视频文件不存在，已尝试: {video_path_in_task_dir}, {video_path_in_files_dir}, {video_path_in_parent}"
                    print(error_msg)
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details(error_msg)
                    return voice_service_pb2.ExtractAudioResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.FAILED,
                        message=error_msg,
                        is_finished=True,
                    )

            # 获取请求参数
            sample_rate = request.sample_rate if request.sample_rate else 44100
            mono = request.mono if hasattr(request, "mono") else True
            start_time = request.start_time if hasattr(request, "start_time") else 0.0
            duration = request.duration if hasattr(request, "duration") else 10.0
            auto_detect_voice = (
                request.auto_detect_voice
                if hasattr(request, "auto_detect_voice")
                else False
            )
            max_silence = (
                request.max_silence if hasattr(request, "max_silence") else 2.0
            )

            # 创建视频处理器
            video_processor = VideoProcessor(output_dir=task_dir)

            try:
                print(f"开始从视频提取音频: {video_name}, 路径: {video_path}")
                output_path = video_processor.extract_audio(
                    video_path=video_path,
                    sample_rate=sample_rate,
                    mono=mono,
                    start_time=start_time,
                    duration=duration,
                    auto_detect_voice=auto_detect_voice,
                    max_silence=max_silence,
                )

                if output_path:
                    # 生成包含任务步骤的输出文件名
                    file_base, file_ext = os.path.splitext(
                        os.path.basename(output_path)
                    )
                    output_filename = f"{file_base}_step{task_step}{file_ext}"

                    # 移动文件到正确的名称
                    new_output_path = os.path.join(task_dir, output_filename)
                    if output_path != new_output_path:
                        os.rename(output_path, new_output_path)

                    print(f"音频提取完成，保存为: {new_output_path}")

                    # 创建语音处理器用于识别
                    processor = VoiceProcessorWithProgress(
                        model_dir=MODEL_DIR,
                        whisper_model_size=WHISPER_SIZE,
                        compute_type=COMPUTE_TYPE,
                        language=LANGUAGE,
                        output_dir=task_dir,
                    )

                    # 使用Whisper识别音频内容
                    recognition_results = processor.transcribe(new_output_path)
                    prompt_text = ""

                    if recognition_results:
                        # 将所有识别结果拼接成一段文字，用中文逗号分隔
                        prompt_text = "，".join(
                            [result["text"] for result in recognition_results]
                        )
                        logger.info(f"音频识别结果: {prompt_text}")

                    return voice_service_pb2.ExtractAudioResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.COMPLETED,
                        message="提取完成",
                        is_finished=True,
                        output_filename=output_filename,
                        prompt_text=prompt_text,  # 添加识别出的文本
                    )
                else:
                    error_msg = "提取失败，处理器未返回有效路径"
                    print(error_msg)
                    return voice_service_pb2.ExtractAudioResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.FAILED,
                        message=error_msg,
                        is_finished=True,
                    )

            except Exception as e:
                error_message = f"提取音频时发生错误: {str(e)}"
                print(error_message)
                return voice_service_pb2.ExtractAudioResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message=error_message,
                    is_finished=True,
                )
            finally:
                # 清理GPU内存
                clean_gpu_memory()

        except Exception as e:
            error_message = f"处理请求时发生错误: {str(e)}"
            print(error_message)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_message)
            return voice_service_pb2.ExtractAudioResponse(
                task_id=task_id if "task_id" in locals() else str(uuid.uuid4()),
                status=voice_service_pb2.TaskStatus.FAILED,
                message=error_message,
                is_finished=True,
            )

    def AddSubtitleToVideo(self, request, context):
        """
        为视频添加智能对齐的硬字幕 - 同步方法

        字幕参数格式说明:

        font_color (字体颜色) 支持格式:
        - 英文颜色名: white, black, red, green, blue, yellow, cyan, magenta, orange, purple
        - 十六进制格式: #FFFFFF, #FF0000, #00FF00, #0000FF, #FFFF00
        - RGB十六进制: 0xFFFFFF, 0xFF0000, 0x00FF00
        - 透明度格式: white@0.8, #FF0000@0.5 (透明度0.0-1.0)
        - 注意: 请勿使用中文颜色名（如"红色"），FFmpeg无法识别

        background_color (背景颜色) 支持格式:
        - 完全相同于font_color的所有格式！
        - 英文颜色名: white, black, red, green, blue, yellow, cyan, magenta, orange, purple
        - 十六进制格式: #FFFFFF, #FF0000, #00FF00, #0000FF, #FFFF00
        - RGB十六进制: 0xFFFFFF, 0xFF0000, 0x00FF00
        - 透明度格式: black@0.5, #000000@0.3, white@0.8 (推荐0.3-0.7)
        - 特殊值: transparent (透明背景), none (无背景)

        font_name (字体名称) 支持:
        - 系统字体: Arial, Times, Helvetica, Verdana, Tahoma
        - Linux字体: DejaVu Sans, Liberation Sans, Noto Sans CJK SC (中文)
        - 字体文件路径: /path/to/font.ttf (如果使用自定义字体)
        - 注意: 中文字体名需要系统支持，建议使用英文字体名

        position (位置) 支持值:
        - bottom: 底部75%-80%区域 (默认，最佳观看体验)
        - top: 顶部居中
        - middle: 画面中央
        - left: 左侧垂直居中
        - right: 右侧垂直居中

        font_size (字体大小):
        - 范围: 8-128 像素
        - 默认值: 32 (自动根据视频分辨率调整)
        - 720p视频推荐: 28-36
        - 1080p视频推荐: 32-48
        - 4K视频推荐: 48-72
        - 注意: 系统会根据视频分辨率自动优化字体大小

        显示效果优化:
        - 自动添加阴影效果提高可读性
        - 智能字体选择，优先支持中文显示
        - 边框宽度增强，确保在各种背景下清晰可见
        - 位置优化，避免遮挡重要画面内容
        """
        try:
            # 获取请求参数
            video_name = request.video_name
            subtitle_text = request.subtitle_text

            # 获取或生成任务ID
            task_id = request.task_id if request.task_id else str(uuid.uuid4())
            task_step = request.task_step if request.task_step else "0"

            logger.info(
                f"收到添加字幕请求 - task_id: {task_id}, video_name: {video_name}, 字幕文本长度: {len(subtitle_text)}"
            )

            # 获取字体样式参数（带格式验证和默认值）

            # 字体名称 - 默认Arial，支持系统字体和字体文件路径
            font_name = (
                request.font_name.strip()
                if hasattr(request, "font_name") and request.font_name
                else "Arial"
            )

            # 字体大小 - 范围8-128，默认32（从24调整为32以提高可读性）
            font_size = (
                max(8, min(128, request.font_size))
                if hasattr(request, "font_size") and request.font_size > 0
                else 32  # 提高默认字体大小
            )

            # 字体颜色 - 支持英文颜色名、十六进制、RGB格式，默认white
            font_color = (
                request.font_color.strip()
                if hasattr(request, "font_color") and request.font_color
                else "white"
            )

            # 背景颜色 - 新增参数，支持透明度，默认transparent
            background_color = (
                request.background_color.strip()
                if hasattr(request, "background_color") and request.background_color
                else "transparent"
            )

            # 边框设置 - 默认true添加黑色边框提高可读性
            add_border = request.add_border if hasattr(request, "add_border") else True

            # 字幕位置 - 支持bottom/top/middle/left/right，默认bottom
            position = (
                request.position.strip().lower()
                if hasattr(request, "position") and request.position
                else "bottom"
            )

            # 验证和记录参数
            logger.info(f"字幕样式参数:")
            logger.info(f"  字体: {font_name} (大小: {font_size})")
            logger.info(f"  颜色: {font_color}")
            logger.info(f"  背景: {background_color}")
            logger.info(f"  边框: {'是' if add_border else '否'}")
            logger.info(f"  位置: {position}")

            # 获取任务专属目录
            task_dir = self._get_task_directory(task_id)

            # 设置视频文件查找路径，首先尝试在任务目录中查找，如果不存在则在FILES_DIR中查找
            video_path_in_task_dir = os.path.join(task_dir, video_name)
            video_path_in_files_dir = os.path.join(FILES_DIR, video_name)

            if os.path.exists(video_path_in_task_dir):
                video_path = video_path_in_task_dir
                logger.info(f"在任务目录中找到视频文件: {video_path}")
            elif os.path.exists(video_path_in_files_dir):
                video_path = video_path_in_files_dir
                logger.info(f"在FILES_DIR中找到视频文件: {video_path}")
            else:
                # 如果都找不到，尝试在task_dir的父目录中查找
                parent_dir = os.path.dirname(task_dir)
                video_path_in_parent = os.path.join(parent_dir, video_name)
                if os.path.exists(video_path_in_parent):
                    video_path = video_path_in_parent
                    logger.info(f"在父目录中找到视频文件: {video_path}")
                else:
                    error_msg = f"视频文件不存在，已尝试: {video_path_in_task_dir}, {video_path_in_files_dir}, {video_path_in_parent}"
                    logger.error(error_msg)
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details(error_msg)
                    return voice_service_pb2.AddSubtitleResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.FAILED,
                        message=error_msg,
                        is_finished=True,
                    )

            # 创建视频字幕处理器
            subtitle_processor = VideoSubtitleProcessor(output_dir=task_dir)

            try:
                logger.info(
                    f"开始为视频添加智能对齐硬字幕: {video_name}, 路径: {video_path}"
                )

                output_path = subtitle_processor.add_subtitle_to_video(
                    video_path=video_path,
                    subtitle_text=subtitle_text,
                    font_name=font_name,
                    font_size=font_size,
                    font_color=font_color,
                    background_color=background_color,  # 新增背景颜色参数
                    add_border=add_border,
                    position=position,
                )

                if output_path:
                    # 生成包含任务步骤的输出文件名
                    file_base, file_ext = os.path.splitext(
                        os.path.basename(output_path)
                    )
                    output_filename = f"{file_base}_step{task_step}{file_ext}"

                    # 移动文件到正确的名称
                    new_output_path = os.path.join(task_dir, output_filename)
                    if output_path != new_output_path:
                        import shutil

                        shutil.move(output_path, new_output_path)

                    logger.info(f"智能硬字幕添加完成，保存为: {new_output_path}")

                    return voice_service_pb2.AddSubtitleResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.COMPLETED,
                        message="智能硬字幕添加完成",
                        is_finished=True,
                        output_filename=output_filename,
                    )
                else:
                    error_msg = "智能硬字幕添加失败，处理器未返回有效路径"
                    logger.error(error_msg)
                    return voice_service_pb2.AddSubtitleResponse(
                        task_id=task_id,
                        status=voice_service_pb2.TaskStatus.FAILED,
                        message=error_msg,
                        is_finished=True,
                    )

            except Exception as e:
                error_message = f"添加智能硬字幕时发生错误: {str(e)}"
                logger.error(error_message)
                return voice_service_pb2.AddSubtitleResponse(
                    task_id=task_id,
                    status=voice_service_pb2.TaskStatus.FAILED,
                    message=error_message,
                    is_finished=True,
                )

        except Exception as e:
            error_message = f"处理请求时发生错误: {str(e)}"
            logger.error(error_message)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_message)
            return voice_service_pb2.AddSubtitleResponse(
                task_id=task_id if "task_id" in locals() else str(uuid.uuid4()),
                status=voice_service_pb2.TaskStatus.FAILED,
                message=error_message,
                is_finished=True,
            )


def serve(port=50051, max_workers=15):
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
    logger.info(f"支持接口: 语音克隆/合成、视频提取音频、智能硬字幕")
    logger.info(f"理论并发能力: 每个接口最多5个任务 (总计{max_workers}个)")
    logger.info(f"消息大小限制: 200MB")
    logger.info(f"连接keepalive: 30秒")
    logger.info(f"连接最大存活时间: 30分钟")
    logger.info(f"文件目录: {FILES_DIR}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在优雅关闭服务...")
        # 优雅关闭，给正在处理的请求60秒完成时间
        server.stop(grace=60)
        logger.info("服务已停止")


if __name__ == "__main__":
    # 从环境变量读取配置
    port = int(os.environ.get("GRPC_PORT", 50051))
    max_workers = int(os.environ.get("GRPC_MAX_WORKERS", 15))

    logger.info(f"从环境变量读取配置: port={port}, max_workers={max_workers}")
    serve(port=port, max_workers=max_workers)
