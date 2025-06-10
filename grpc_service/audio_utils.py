#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频处理工具类，提供音频裁剪等功能
"""

import os
import logging
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Union, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("audio_utils")


def trim_audio(
    input_path: str,
    output_path: Optional[str] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    output_format: str = "wav",
) -> str:
    """
    裁剪音频文件

    Args:
        input_path: 输入音频文件路径
        output_path: 输出音频文件路径，如果为None则自动生成
        start_time: 开始时间（秒）
        end_time: 结束时间（秒），如果为None则截取到结尾
        output_format: 输出格式

    Returns:
        裁剪后的音频文件路径
    """
    if not os.path.exists(input_path):
        logger.error(f"输入音频文件不存在: {input_path}")
        return ""

    try:
        # 读取音频文件
        data, sample_rate = sf.read(input_path)

        # 计算起始和结束样本
        start_sample = int(start_time * sample_rate)

        if end_time is not None:
            end_sample = int(end_time * sample_rate)
        else:
            end_sample = len(data)

        # 验证范围有效性
        if start_sample >= len(data):
            logger.error(f"起始时间 {start_time}秒 超出音频长度")
            return ""

        if end_sample > len(data):
            logger.warning(f"结束时间超出音频长度，将截取到结尾")
            end_sample = len(data)

        if start_sample >= end_sample:
            logger.error(f"起始时间 {start_time}秒 大于或等于结束时间 {end_time}秒")
            return ""

        # 裁剪音频
        trimmed_data = data[start_sample:end_sample]

        # 创建输出路径
        if output_path is None:
            dir_name = os.path.dirname(input_path)
            base_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(base_name)[0]

            # 修复格式化问题
            if end_time is None:
                end_str = "end"
            else:
                end_str = f"{end_time:.1f}"

            output_path = os.path.join(
                dir_name,
                f"{name_without_ext}_trimmed_{start_time:.1f}s-{end_str}s.{output_format}",
            )

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存裁剪后的音频
        sf.write(output_path, trimmed_data, sample_rate)

        duration = len(trimmed_data) / sample_rate
        logger.info(f"音频裁剪成功: {output_path}, 长度: {duration:.2f}秒")

        return output_path

    except Exception as e:
        logger.exception(f"裁剪音频时发生错误: {e}")
        return ""


def concat_audio(
    input_paths: list,
    output_path: Optional[str] = None,
    add_silence: float = 0.5,
    output_format: str = "wav",
) -> str:
    """
    拼接多个音频文件

    Args:
        input_paths: 输入音频文件路径列表
        output_path: 输出音频文件路径，如果为None则自动生成
        add_silence: 音频之间添加的静音长度（秒）
        output_format: 输出格式

    Returns:
        拼接后的音频文件路径
    """
    if not input_paths:
        logger.error("输入音频文件列表为空")
        return ""

    try:
        all_data = []
        sample_rate = None

        # 读取所有音频文件
        for i, path in enumerate(input_paths):
            if not os.path.exists(path):
                logger.error(f"音频文件不存在: {path}")
                return ""

            data, rate = sf.read(path)

            # 确保所有音频的采样率一致
            if sample_rate is None:
                sample_rate = rate
            elif rate != sample_rate:
                logger.warning(
                    f"文件 {path} 采样率({rate})与第一个文件采样率({sample_rate})不一致，可能导致问题"
                )

            # 如果不是第一个文件，添加静音
            if i > 0 and add_silence > 0:
                silence = np.zeros(int(add_silence * sample_rate), dtype=data.dtype)
                all_data.append(silence)

            all_data.append(data)

        # 拼接音频
        concatenated_data = np.concatenate(all_data)

        # 创建输出路径
        if output_path is None:
            dir_name = os.path.dirname(input_paths[0])
            output_path = os.path.join(
                dir_name, f"concatenated_{len(input_paths)}_files.{output_format}"
            )

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存拼接后的音频
        sf.write(output_path, concatenated_data, sample_rate)

        duration = len(concatenated_data) / sample_rate
        logger.info(f"音频拼接成功: {output_path}, 长度: {duration:.2f}秒")

        return output_path

    except Exception as e:
        logger.exception(f"拼接音频时发生错误: {e}")
        return ""


def trim_voice_section(
    input_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0.02,
    max_silence: float = 2.0,
    min_voice_duration: float = 1.0,
    output_format: str = "wav",
) -> str:
    """
    自动检测并提取音频中的有声部分

    Args:
        input_path: 输入音频文件路径
        output_path: 输出音频文件路径，如果为None则自动生成
        threshold: 音量阈值，低于此值被视为静音，默认0.02
        max_silence: 最大静音时长（秒），超过此值将分割音频片段，默认2秒
        min_voice_duration: 最小有声片段长度（秒），小于此值的片段将被忽略，默认1秒
        output_format: 输出格式

    Returns:
        提取有声部分后的音频文件路径
    """
    if not os.path.exists(input_path):
        logger.error(f"输入音频文件不存在: {input_path}")
        return ""

    try:
        # 读取音频文件
        data, sample_rate = sf.read(input_path)

        # 如果是立体声，转为单声道用于语音检测
        if len(data.shape) > 1 and data.shape[1] > 1:
            mono_data = np.mean(data, axis=1)
        else:
            mono_data = data

        # 计算音量包络
        frame_size = int(0.02 * sample_rate)  # 20ms帧
        hop_size = frame_size // 2  # 50%重叠

        frame_count = (len(mono_data) - frame_size) // hop_size + 1
        energy = np.zeros(frame_count)

        for i in range(frame_count):
            start = i * hop_size
            end = start + frame_size
            frame = mono_data[start:end]
            energy[i] = np.sqrt(np.mean(frame**2))

        # 应用阈值，标记有声帧
        is_voice = energy > threshold

        # 计算每帧对应的时间点
        frame_times = np.arange(frame_count) * hop_size / sample_rate

        # 转换为连续的语音片段
        voice_segments = []
        in_voice = False
        start_time = 0

        for i, (is_v, time) in enumerate(zip(is_voice, frame_times)):
            if is_v and not in_voice:  # 开始有声音
                in_voice = True
                start_time = time
            elif not is_v and in_voice:  # 结束有声音
                # 检查静音时长
                silence_duration = 0
                j = i
                while j < len(is_voice) and not is_voice[j]:
                    silence_duration += hop_size / sample_rate
                    j += 1

                # 如果静音时长超过阈值，或者是最后一帧，则切分
                if silence_duration >= max_silence or j >= len(is_voice):
                    end_time = time
                    duration = end_time - start_time
                    # 只保留长度足够的片段
                    if duration >= min_voice_duration:
                        voice_segments.append((start_time, end_time))
                    in_voice = False

        # 处理最后一个片段
        if in_voice:
            end_time = frame_times[-1]
            duration = end_time - start_time
            if duration >= min_voice_duration:
                voice_segments.append((start_time, end_time))

        # 如果没有检测到有效的语音片段，返回原始音频
        if not voice_segments:
            logger.warning(f"未检测到有效的语音片段，返回原始音频: {input_path}")
            return input_path

        logger.info(f"检测到 {len(voice_segments)} 个语音片段")

        # 选择最长的语音片段
        longest_segment = max(voice_segments, key=lambda x: x[1] - x[0])
        logger.info(
            f"选择最长语音片段: {longest_segment[0]:.2f}s - {longest_segment[1]:.2f}s"
        )

        # 添加前后缓冲
        buffer = 0.2  # 200毫秒
        start_time = max(0, longest_segment[0] - buffer)
        end_time = min(len(data) / sample_rate, longest_segment[1] + buffer)

        # 裁剪音频
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        trimmed_data = data[start_sample:end_sample]

        # 创建输出路径
        if output_path is None:
            dir_name = os.path.dirname(input_path)
            base_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(
                dir_name,
                f"{name_without_ext}_voice_{start_time:.1f}s-{end_time:.1f}s.{output_format}",
            )

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存裁剪后的音频
        sf.write(output_path, trimmed_data, sample_rate)

        duration = len(trimmed_data) / sample_rate
        logger.info(f"语音提取成功: {output_path}, 长度: {duration:.2f}秒")

        return output_path

    except Exception as e:
        logger.exception(f"提取语音部分时发生错误: {e}")
        return ""


def check_audio_silence(
    input_path: str,
    threshold: float = 0.01,
    min_voice_ratio: float = 0.05,
) -> Tuple[bool, float]:
    """
    检查音频是否全部为静音或有声部分过少

    Args:
        input_path: 输入音频文件路径
        threshold: 音量阈值，低于此值被视为静音，默认0.01
        min_voice_ratio: 最小有声比例，低于此比例视为无效音频，默认5%

    Returns:
        元组 (is_silent, voice_ratio)
        - is_silent: 是否为静音或无效音频
        - voice_ratio: 有声部分占总时长的比例
    """
    if not os.path.exists(input_path):
        logger.error(f"输入音频文件不存在: {input_path}")
        return True, 0.0

    try:
        # 读取音频文件
        data, sample_rate = sf.read(input_path)

        # 如果是立体声，转为单声道用于检测
        if len(data.shape) > 1 and data.shape[1] > 1:
            mono_data = np.mean(data, axis=1)
        else:
            mono_data = data

        # 计算音量包络
        frame_size = int(0.02 * sample_rate)  # 20ms帧
        hop_size = frame_size // 2  # 50%重叠

        frame_count = (len(mono_data) - frame_size) // hop_size + 1
        energy = np.zeros(frame_count)

        for i in range(frame_count):
            start = i * hop_size
            end = start + frame_size
            if end <= len(mono_data):
                frame = mono_data[start:end]
                energy[i] = np.sqrt(np.mean(frame**2))

        # 计算有声帧的比例
        voice_frames = np.sum(energy > threshold)
        total_frames = len(energy)
        voice_ratio = voice_frames / total_frames if total_frames > 0 else 0.0

        # 判断是否为静音或无效音频
        is_silent = voice_ratio < min_voice_ratio

        logger.info(f"音频检测结果: 有声比例={voice_ratio:.2%}, 是否静音={is_silent}")
        return is_silent, voice_ratio

    except Exception as e:
        logger.exception(f"检测音频静音时发生错误: {e}")
        return True, 0.0


def remove_silence_from_ends(
    input_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0.01,
    buffer_duration: float = 0.1,
    output_format: str = "wav",
) -> str:
    """
    移除音频前后的静音片段

    Args:
        input_path: 输入音频文件路径
        output_path: 输出音频文件路径，如果为None则覆盖原文件
        threshold: 音量阈值，低于此值被视为静音，默认0.01
        buffer_duration: 保留的缓冲时间（秒），默认0.1秒
        output_format: 输出格式

    Returns:
        处理后的音频文件路径，如果失败返回原路径
    """
    if not os.path.exists(input_path):
        logger.error(f"输入音频文件不存在: {input_path}")
        return input_path

    try:
        # 读取音频文件
        data, sample_rate = sf.read(input_path)
        original_duration = len(data) / sample_rate

        # 如果是立体声，转为单声道用于检测
        if len(data.shape) > 1 and data.shape[1] > 1:
            mono_data = np.mean(data, axis=1)
        else:
            mono_data = data

        # 计算音量包络
        frame_size = int(0.02 * sample_rate)  # 20ms帧
        hop_size = frame_size // 2  # 50%重叠

        frame_count = (len(mono_data) - frame_size) // hop_size + 1
        energy = np.zeros(frame_count)

        for i in range(frame_count):
            start = i * hop_size
            end = start + frame_size
            if end <= len(mono_data):
                frame = mono_data[start:end]
                energy[i] = np.sqrt(np.mean(frame**2))

        # 找到第一个和最后一个有声帧
        voice_frames = energy > threshold

        if not np.any(voice_frames):
            logger.warning(f"音频全部为静音，返回原文件: {input_path}")
            return input_path

        first_voice_frame = np.argmax(voice_frames)
        last_voice_frame = len(voice_frames) - 1 - np.argmax(voice_frames[::-1])

        # 转换为样本索引，添加缓冲
        buffer_samples = int(buffer_duration * sample_rate)

        start_sample = max(0, first_voice_frame * hop_size - buffer_samples)
        end_sample = min(
            len(data), (last_voice_frame + 1) * hop_size + frame_size + buffer_samples
        )

        # 裁剪音频
        trimmed_data = data[start_sample:end_sample]
        new_duration = len(trimmed_data) / sample_rate

        # 如果裁剪后的音频太短，返回原文件
        if new_duration < 0.5:  # 少于0.5秒
            logger.warning(f"裁剪后音频过短({new_duration:.2f}s)，返回原文件")
            return input_path

        # 创建输出路径
        if output_path is None:
            output_path = input_path  # 覆盖原文件

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存裁剪后的音频
        sf.write(output_path, trimmed_data, sample_rate)

        logger.info(f"静音移除成功: {output_path}")
        logger.info(
            f"原始时长: {original_duration:.2f}s -> 新时长: {new_duration:.2f}s"
        )
        logger.info(
            f"移除了前面 {start_sample/sample_rate:.2f}s 和后面 {(len(data)-end_sample)/sample_rate:.2f}s"
        )

        return output_path

    except Exception as e:
        logger.exception(f"移除静音时发生错误: {e}")
        return input_path


if __name__ == "__main__":
    # 测试裁剪功能
    input_audio = "../outputs/voice_clone_20250412_073310_bdf7e5fe.wav"
    if os.path.exists(input_audio):
        # 裁剪前5秒
        trim_audio(input_audio, start_time=0, end_time=5)
        # 裁剪从10秒到15秒的部分
        trim_audio(input_audio, start_time=10, end_time=15)
        # 裁剪从20秒到结束的部分
        trim_audio(input_audio, start_time=20)
        # 测试语音提取功能
        trim_voice_section(input_audio)
    else:
        print(f"测试音频文件不存在: {input_audio}")
