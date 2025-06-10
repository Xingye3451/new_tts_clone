#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import uuid
import tempfile
import json
import logging
import shutil
import re
import wave
from typing import Dict, List, Optional, Tuple, Callable
from difflib import SequenceMatcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("video_processor.log")],
)
logger = logging.getLogger("video_subtitle_processor")


class VideoSubtitleProcessor:
    """视频字幕处理器，用于为视频添加智能对齐的硬字幕"""

    def __init__(self, output_dir: str = "outputs"):
        """
        初始化视频字幕处理器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 检查ffmpeg是否可用
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, check=True
            )
            logger.info(f"ffmpeg 可用: {result.stdout.splitlines()[0]}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"ffmpeg 检测失败: {e}")
            raise RuntimeError("ffmpeg 不可用，请确保已正确安装")

    def _extract_audio_from_video(self, video_path: str) -> str:
        """
        从视频中提取音频，用于语音识别

        Args:
            video_path: 视频文件路径

        Returns:
            音频文件路径
        """
        # 创建临时音频文件
        audio_path = os.path.join(
            self.output_dir, f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        )

        try:
            # 使用ffmpeg从视频中提取音频
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",  # 禁用视频
                "-acodec",
                "pcm_s16le",  # 使用无损编码
                "-ar",
                "16000",  # 采样率16kHz，适合语音识别
                "-ac",
                "1",  # 单声道
                audio_path,
            ]

            logger.info(f"从视频提取音频: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)

            return audio_path
        except Exception as e:
            logger.error(f"从视频提取音频失败: {e}")
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            raise

    def _transcribe_audio_with_whisper(self, audio_path: str) -> List[Dict]:
        """
        使用Whisper进行语音识别，获取带时间戳的转录结果

        Args:
            audio_path: 音频文件路径

        Returns:
            转录结果列表，每个元素包含 {'text': str, 'start': float, 'end': float}
        """
        try:
            # 导入语音处理器（与其他接口保持一致）
            import sys

            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from voice_processor_wrapper import VoiceProcessorWithProgress

            # 获取环境变量（与其他接口保持一致）
            MODEL_DIR = os.environ.get("MODEL_DIR", "pretrained_models/Spark-TTS-0.5B")
            WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")
            COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
            LANGUAGE = os.environ.get("LANGUAGE", "zh")

            # 创建语音处理器（与其他接口保持一致）
            processor = VoiceProcessorWithProgress(
                model_dir=MODEL_DIR,
                whisper_model_size=WHISPER_SIZE,
                compute_type=COMPUTE_TYPE,
                language=LANGUAGE,
                output_dir=self.output_dir,
            )

            # 进行语音识别（使用与其他接口相同的方法）
            logger.info(f"开始使用Whisper识别音频: {audio_path}")
            recognition_results = processor.transcribe(audio_path)

            if recognition_results:
                logger.info(
                    f"Whisper识别完成，共识别到 {len(recognition_results)} 个片段"
                )
                for i, result in enumerate(recognition_results):
                    logger.info(
                        f"片段 {i+1}: [{result['start']:.1f}s-{result['end']:.1f}s] {result['text']}"
                    )
                return recognition_results
            else:
                logger.warning("Whisper识别未返回结果")
                return []

        except Exception as e:
            logger.error(f"Whisper语音识别失败: {e}")
            return []

    def _clean_text_for_alignment(self, text: str) -> str:
        """
        清理文本用于对齐，去除标点符号和多余空白

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        import re

        # 去除所有标点符号和特殊字符，只保留中文、英文、数字
        cleaned = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf\w\s]", "", text)

        # 去除多余空白
        cleaned = re.sub(r"\s+", "", cleaned)

        return cleaned.strip()

    def _align_text_with_recognition(
        self, user_text: str, recognition_results: List[Dict]
    ) -> List[Dict]:
        """
        将用户提供的准确文本与Whisper识别结果进行对齐
        优化版本：专注于精确同步

        Args:
            user_text: 用户提供的准确文本
            recognition_results: Whisper识别结果

        Returns:
            对齐后的字幕片段列表
        """
        if not recognition_results:
            logger.warning("没有识别结果，使用简单分段方式")
            return self._simple_text_segmentation(user_text)

        logger.info("使用优化的智能对齐算法")

        # 获取语音的实际时间范围
        speech_start = recognition_results[0]["start"]
        speech_end = recognition_results[-1]["end"]
        total_duration = speech_end - speech_start

        logger.info(
            f"语音时间范围: {speech_start:.2f}s - {speech_end:.2f}s (总时长: {total_duration:.2f}s)"
        )

        # 清理和预处理文本
        cleaned_user_text = self._clean_text_for_alignment(user_text)

        # 合并识别结果的文本
        recognition_text = ""
        for result in recognition_results:
            recognition_text += result["text"]
        cleaned_recognition_text = self._clean_text_for_alignment(recognition_text)

        logger.info(
            f"用户文本长度: {len(cleaned_user_text)}, 识别文本长度: {len(cleaned_recognition_text)}"
        )

        # 计算文本相似度，决定对齐策略
        similarity = self._calculate_text_similarity(
            cleaned_user_text, cleaned_recognition_text
        )
        logger.info(f"文本相似度: {similarity:.2f}")

        if similarity > 0.7:
            # 高相似度：使用精确匹配对齐
            aligned_segments = self._precise_matching_alignment(
                user_text, recognition_results
            )
        elif similarity > 0.4:
            # 中等相似度：使用智能分段对齐
            aligned_segments = self._smart_segmentation_alignment(
                user_text, recognition_results
            )
        else:
            # 低相似度：使用简单时间比例对齐
            aligned_segments = self._simple_time_alignment(
                user_text, recognition_results
            )

        # 修复重叠问题
        fixed_segments = self._fix_subtitle_overlaps(aligned_segments)

        return fixed_segments

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0

        # 使用SequenceMatcher计算相似度
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _precise_matching_alignment(
        self, user_text: str, recognition_results: List[Dict]
    ) -> List[Dict]:
        """
        精确匹配对齐（高相似度时使用）
        """
        logger.info("使用精确匹配对齐")

        # 智能分段用户文本
        user_segments = self._intelligent_text_segmentation(user_text)

        # 为每个用户段落找到最佳匹配的识别段落
        aligned_segments = []

        for i, user_segment in enumerate(user_segments):
            # 计算在整个文本中的位置比例
            total_chars = sum(len(seg) for seg in user_segments)
            chars_before = sum(len(user_segments[j]) for j in range(i))
            start_ratio = chars_before / total_chars if total_chars > 0 else 0
            end_ratio = (
                (chars_before + len(user_segment)) / total_chars
                if total_chars > 0
                else 1
            )

            # 根据比例映射到识别结果的时间
            speech_start = recognition_results[0]["start"]
            speech_end = recognition_results[-1]["end"]
            total_duration = speech_end - speech_start

            segment_start = speech_start + start_ratio * total_duration
            segment_end = speech_start + end_ratio * total_duration

            # 优化显示时间计算
            # 最小时间：每秒6个字符（提高阅读速度）
            # 最大时间：每个字符0.3秒（避免显示过久）
            min_duration = max(0.8, len(user_segment) / 6.0)  # 每秒6个字符，最少0.8秒
            max_duration = min(4.0, len(user_segment) * 0.3)  # 每个字符0.3秒，最多4秒

            # 确保在合理范围内
            calculated_duration = segment_end - segment_start
            if calculated_duration < min_duration:
                segment_end = segment_start + min_duration
            elif calculated_duration > max_duration:
                segment_end = segment_start + max_duration

            aligned_segments.append(
                {"text": user_segment, "start": segment_start, "end": segment_end}
            )

            logger.info(
                f"精确对齐片段 {i+1}: [{segment_start:.1f}s-{segment_end:.1f}s] ({segment_end-segment_start:.1f}s) {user_segment[:30]}..."
            )

        return aligned_segments

    def _smart_segmentation_alignment(
        self, user_text: str, recognition_results: List[Dict]
    ) -> List[Dict]:
        """
        智能分段对齐（中等相似度时使用）
        """
        logger.info("使用智能分段对齐")

        # 基于识别结果的时间点进行智能分段
        user_segments = self._intelligent_text_segmentation(user_text)
        recognition_segments = recognition_results

        aligned_segments = []

        # 如果用户段落数量接近识别段落数量，尝试一对一映射
        if abs(len(user_segments) - len(recognition_segments)) <= 2:
            for i, user_segment in enumerate(user_segments):
                if i < len(recognition_segments):
                    # 使用对应的识别段落时间
                    rec_segment = recognition_segments[i]
                    segment_start = rec_segment["start"]
                    segment_end = rec_segment["end"]
                else:
                    # 超出识别段落，使用最后一个段落的时间延伸
                    last_rec = recognition_segments[-1]
                    segment_start = last_rec["end"]
                    segment_end = segment_start + 1.5  # 减少默认时间从2秒到1.5秒

                # 优化显示时间
                min_duration = max(0.8, len(user_segment) / 6.0)  # 每秒6个字符
                max_duration = min(
                    3.0, len(user_segment) * 0.25
                )  # 每个字符0.25秒，最多3秒

                calculated_duration = segment_end - segment_start
                if calculated_duration < min_duration:
                    segment_end = segment_start + min_duration
                elif calculated_duration > max_duration:
                    segment_end = segment_start + max_duration

                aligned_segments.append(
                    {"text": user_segment, "start": segment_start, "end": segment_end}
                )
        else:
            # 段落数量差异较大，使用时间比例分配
            return self._simple_time_alignment(user_text, recognition_results)

        return aligned_segments

    def _simple_time_alignment(
        self, user_text: str, recognition_results: List[Dict]
    ) -> List[Dict]:
        """
        简单时间比例对齐（低相似度或备用方案）
        """
        logger.info("使用简单时间比例对齐")

        speech_start = recognition_results[0]["start"]
        speech_end = recognition_results[-1]["end"]
        total_duration = speech_end - speech_start

        # 智能分段
        user_segments = self._intelligent_text_segmentation(user_text)

        aligned_segments = []
        segment_count = len(user_segments)

        for i, segment_text in enumerate(user_segments):
            # 时间比例分配
            start_ratio = i / segment_count
            end_ratio = (i + 1) / segment_count

            segment_start = speech_start + start_ratio * total_duration
            segment_end = speech_start + end_ratio * total_duration

            # 优化显示时间
            min_duration = max(0.8, len(segment_text) / 6.0)  # 每秒6个字符
            max_duration = min(3.0, len(segment_text) * 0.25)  # 每个字符0.25秒

            calculated_duration = segment_end - segment_start
            if calculated_duration < min_duration:
                segment_end = segment_start + min_duration
                if segment_end > speech_end:
                    segment_end = speech_end
                    segment_start = max(speech_start, segment_end - min_duration)
            elif calculated_duration > max_duration:
                segment_end = segment_start + max_duration

            aligned_segments.append(
                {"text": segment_text, "start": segment_start, "end": segment_end}
            )

        return aligned_segments

    def _intelligent_text_segmentation(self, text: str) -> List[str]:
        """
        智能文本分段，优化版本
        """
        import re

        # 清理文本
        text = re.sub(r"\s+", " ", text).strip()

        # 第一级分段：按句号、问号、感叹号
        primary_segments = re.split(r"([。！？\.!?]+)", text)

        # 合并标点符号
        segments = []
        for i in range(0, len(primary_segments) - 1, 2):
            if i + 1 < len(primary_segments):
                segment = primary_segments[i] + primary_segments[i + 1]
                if segment.strip():
                    segments.append(segment.strip())
            elif primary_segments[i].strip():
                segments.append(primary_segments[i].strip())

        # 处理最后一个片段
        if len(primary_segments) % 2 == 1 and primary_segments[-1].strip():
            segments.append(primary_segments[-1].strip())

        # 如果分段太少，进行二级分段
        if len(segments) <= 2 and len(text) > 40:
            new_segments = []
            for segment in segments:
                # 按逗号、分号分段
                parts = re.split(r"([，,；;、]+)", segment)
                temp_parts = []
                for j in range(0, len(parts) - 1, 2):
                    if j + 1 < len(parts):
                        part = parts[j] + parts[j + 1]
                        if part.strip():
                            temp_parts.append(part.strip())
                    elif parts[j].strip():
                        temp_parts.append(parts[j].strip())

                if len(parts) % 2 == 1 and parts[-1].strip():
                    temp_parts.append(parts[-1].strip())

                new_segments.extend(temp_parts if temp_parts else [segment])

            if new_segments:
                segments = new_segments

        # 长度控制：确保每个段落不超过30个字符
        final_segments = []
        for segment in segments:
            if len(segment) <= 30:
                final_segments.append(segment)
            else:
                # 长段落强制分割
                words = segment.split()
                current_part = ""
                for word in words:
                    if len(current_part + word) <= 30:
                        current_part += word
                    else:
                        if current_part:
                            final_segments.append(current_part.strip())
                        current_part = word
                if current_part:
                    final_segments.append(current_part.strip())

        # 确保至少有一个段落
        if not final_segments:
            final_segments = [text]

        logger.info(f"智能分段完成: {len(final_segments)} 个片段")
        for i, seg in enumerate(final_segments):
            logger.info(f"  片段 {i+1}: {seg[:50]}...")

        return final_segments

    def _fix_subtitle_overlaps(self, segments: List[Dict]) -> List[Dict]:
        """
        修复字幕片段时间重叠问题

        Args:
            segments: 原始字幕片段

        Returns:
            修复后的字幕片段
        """
        if not segments:
            return segments

        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        fixed_segments = []

        for i, segment in enumerate(sorted_segments):
            current_start = segment["start"]
            current_end = segment["end"]
            current_text = segment["text"]

            # 检查与前一个片段是否重叠
            if fixed_segments:
                prev_segment = fixed_segments[-1]
                if current_start < prev_segment["end"]:
                    # 有重叠，调整当前片段的开始时间
                    gap = 0.05  # 减少间隔从0.1秒到0.05秒
                    current_start = prev_segment["end"] + gap

                    # 重新计算合理的显示时间
                    min_duration = max(0.8, len(current_text) / 6.0)  # 每秒6个字符
                    max_duration = min(3.0, len(current_text) * 0.25)  # 每个字符0.25秒

                    # 优先保证最小时间，但不超过最大时间
                    if current_end - current_start < min_duration:
                        current_end = current_start + min_duration
                    elif current_end - current_start > max_duration:
                        current_end = current_start + max_duration

                    logger.info(
                        f"修复重叠: 片段{i+1}时间调整为 [{current_start:.1f}s-{current_end:.1f}s] ({current_end-current_start:.1f}s)"
                    )

            # 检查与下一个片段是否会重叠
            if i < len(sorted_segments) - 1:
                next_segment = sorted_segments[i + 1]
                if current_end > next_segment["start"]:
                    # 会重叠，调整当前片段的结束时间
                    gap = 0.05  # 减少间隔
                    available_time = next_segment["start"] - current_start - gap

                    # 计算合理的显示时间
                    min_duration = max(0.8, len(current_text) / 6.0)
                    max_duration = min(3.0, len(current_text) * 0.25)

                    # 在可用时间内选择合适的结束时间
                    if available_time >= min_duration:
                        # 有足够空间，使用合理时长
                        optimal_duration = min(max_duration, available_time)
                        current_end = current_start + optimal_duration
                    else:
                        # 空间不足，压缩显示时间
                        current_end = current_start + max(
                            0.5, available_time
                        )  # 最少0.5秒

                        # 如果还是不够，调整开始时间
                        if current_end > next_segment["start"] - gap:
                            current_end = next_segment["start"] - gap
                            current_start = current_end - max(0.5, min_duration)
                            if current_start < 0:
                                current_start = 0
                                current_end = max(0.5, min_duration)

                    logger.info(
                        f"修复重叠: 片段{i+1}时间调整为 [{current_start:.1f}s-{current_end:.1f}s] ({current_end-current_start:.1f}s)"
                    )

            fixed_segments.append(
                {"text": current_text, "start": current_start, "end": current_end}
            )

        logger.info(f"字幕重叠修复完成，共处理 {len(fixed_segments)} 个片段")
        return fixed_segments

    def _simple_text_segmentation_for_alignment(self, text: str) -> List[str]:
        """
        简单的文本分段，专门用于对齐

        Args:
            text: 原始文本

        Returns:
            分段后的文本列表
        """
        import re

        # 清理文本
        text = re.sub(r"\s+", " ", text).strip()

        # 按标点符号分段
        segments = re.split(r"([。！？\.!?]+)", text)

        # 合并分隔符
        result_segments = []
        for i in range(0, len(segments) - 1, 2):
            if i + 1 < len(segments):
                segment = segments[i] + segments[i + 1]
                if segment.strip():
                    result_segments.append(segment.strip())
            elif segments[i].strip():
                result_segments.append(segments[i].strip())

        # 处理最后一个片段
        if len(segments) % 2 == 1 and segments[-1].strip():
            result_segments.append(segments[-1].strip())

        # 如果没有分段或分段太少，按逗号分段
        if len(result_segments) <= 1 and len(text) > 20:
            segments = re.split(r"([，,；;、]+)", text)
            result_segments = []
            for i in range(0, len(segments) - 1, 2):
                if i + 1 < len(segments):
                    segment = segments[i] + segments[i + 1]
                    if segment.strip():
                        result_segments.append(segment.strip())
                elif segments[i].strip():
                    result_segments.append(segments[i].strip())

            if len(segments) % 2 == 1 and segments[-1].strip():
                result_segments.append(segments[-1].strip())

        # 如果还是太少，按长度强制分段
        if len(result_segments) <= 1 and len(text) > 30:
            max_length = 25
            result_segments = []
            current_pos = 0

            while current_pos < len(text):
                end_pos = min(current_pos + max_length, len(text))

                # 尝试在合适位置断开
                if end_pos < len(text):
                    for i in range(end_pos, max(current_pos + 10, end_pos - 10), -1):
                        if text[i] in " ，,的了在是有":
                            end_pos = i + 1
                            break

                segment = text[current_pos:end_pos].strip()
                if segment:
                    result_segments.append(segment)

                current_pos = end_pos

        # 确保至少有一个段落
        if not result_segments:
            result_segments = [text]

        logger.info(f"文本分段完成: {len(result_segments)} 个片段")
        return result_segments

    def _simple_text_segmentation(self, text: str) -> List[Dict]:
        """
        简单文本分段（备用方案）

        Args:
            text: 原始文本

        Returns:
            简单分段的字幕片段
        """
        import re

        # 按句号、问号、感叹号分割
        segments = re.split(r"[。！？\.\!\?]+", text)
        segments = [seg.strip() for seg in segments if seg.strip()]

        if not segments:
            segments = [text]

        # 如果段落太少，按逗号分割
        if len(segments) <= 2 and len(text) > 30:
            new_segments = []
            for segment in segments:
                parts = re.split(r"[，,；;]", segment)
                parts = [part.strip() for part in parts if part.strip()]
                new_segments.extend(parts)
            if new_segments:
                segments = new_segments

        # 生成简单的时间分配（假设10秒总时长）
        total_duration = 10.0
        segment_duration = total_duration / len(segments)

        result = []
        for i, segment in enumerate(segments):
            result.append(
                {
                    "text": segment,
                    "start": i * segment_duration,
                    "end": (i + 1) * segment_duration,
                }
            )

        return result

    def _create_hard_subtitle_video(
        self,
        video_path: str,
        subtitle_segments: List[Dict],
        font_name: str = "Arial",
        font_size: int = 24,
        font_color: str = "white",
        background_color: str = "transparent",
        add_border: bool = True,
        position: str = "bottom",
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        使用FFmpeg创建硬字幕视频

        Args:
            video_path: 原视频路径
            subtitle_segments: 字幕片段列表
            font_name: 字体名称
            font_size: 字体大小
            font_color: 字体颜色
                支持格式:
                - 英文颜色名: white, black, red, green, blue, yellow, cyan, magenta
                - 十六进制: #FFFFFF, #FF0000, #00FF00, #0000FF
                - RGB值: 0xFFFFFF, 0xFF0000
                - 透明度格式: white@0.8, #FF0000@0.5 (透明度0.0-1.0)
            background_color: 背景颜色
                支持格式完全相同于font_color:
                - 英文颜色名: white, black, red, green, blue, yellow, cyan, magenta
                - 十六进制: #FFFFFF, #FF0000, #00FF00, #0000FF
                - RGB值: 0xFFFFFF, 0xFF0000
                - 透明度格式: black@0.5, #000000@0.3 (推荐0.3-0.7)
                - 特殊值: transparent (透明背景), none (无背景)
            add_border: 是否添加边框
            position: 字幕位置 (bottom/top/middle/left/right)
            progress_callback: 进度回调

        Returns:
            输出视频路径
        """
        # 生成输出文件路径
        filename, ext = os.path.splitext(os.path.basename(video_path))
        output_filename = f"{filename}_hard_subtitled_{uuid.uuid4().hex[:8]}{ext}"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            # 获取视频分辨率以动态调整字体大小
            try:
                probe_cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-select_streams",
                    "v:0",
                    video_path,
                ]
                probe_result = subprocess.run(
                    probe_cmd, capture_output=True, text=True, check=True
                )
                video_info = json.loads(probe_result.stdout)
                video_width = int(video_info["streams"][0]["width"])
                video_height = int(video_info["streams"][0]["height"])

                # 根据视频分辨率动态调整字体大小
                if font_size <= 24:  # 只有当用户没有明确指定大字体时才自动调整
                    if video_height >= 1080:
                        font_size = max(font_size, 36)  # 1080p及以上使用36号字体
                    elif video_height >= 720:
                        font_size = max(font_size, 32)  # 720p使用32号字体
                    else:
                        font_size = max(font_size, 28)  # 其他分辨率使用28号字体

                logger.info(
                    f"视频分辨率: {video_width}x{video_height}, 调整字体大小为: {font_size}"
                )

            except Exception as e:
                logger.warning(f"无法获取视频分辨率，使用默认字体大小: {e}")
                # 如果无法获取分辨率，至少将默认字体调大一些
                if font_size <= 24:
                    font_size = 32
                video_height = 720  # 假设默认分辨率

            # 构建drawtext滤镜
            drawtext_filters = []

            for i, segment in enumerate(subtitle_segments):
                text = segment["text"].replace("'", "\\'").replace(":", "\\:")
                start_time = segment["start"]
                end_time = segment["end"]

                # 优化字体路径设置，优先使用支持中文的字体
                font_paths = [
                    # 优先使用支持中文的字体
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/TTF/DejaVuSans.ttf",
                    "/System/Library/Fonts/Arial.ttf",
                    "/Windows/Fonts/arial.ttf",
                    # 如果都找不到，使用系统默认
                    "Arial",
                ]

                font_file = None
                for path in font_paths:
                    if path == "Arial" or os.path.exists(path):
                        font_file = path
                        break

                # 优化位置设置 - 解决位置过低的问题
                if position.lower() == "top":
                    y_pos = "text_h+20"  # 顶部留20像素边距
                elif position.lower() == "middle":
                    y_pos = "(h-text_h)/2"
                elif position.lower() == "left":
                    x_pos = "20"  # 左侧留20像素边距
                    y_pos = "(h-text_h)/2"
                elif position.lower() == "right":
                    x_pos = "w-text_w-20"  # 右侧留20像素边距
                    y_pos = "(h-text_h)/2"
                else:  # bottom - 优化底部位置到75%-80%区域
                    # 计算75%-80%位置：h*0.75 到 h*0.8 之间
                    # 使用h*0.78作为基准位置（78%处）
                    y_pos = f"h*0.78-text_h/2"  # 字幕中心位置在78%处

                # 构建drawtext参数
                drawtext_params = [
                    f"text='{text}'",
                    f"enable='between(t,{start_time},{end_time})'",
                    f"fontsize={font_size}",
                    f"fontcolor={font_color}",
                ]

                # 设置x坐标（水平位置）
                if position.lower() in ["left", "right"]:
                    drawtext_params.append(f"x={x_pos}")
                else:
                    drawtext_params.append("x=(w-text_w)/2")  # 水平居中

                # 设置y坐标（垂直位置）
                drawtext_params.append(f"y={y_pos}")

                # 字体文件设置 - 确保字体正确加载
                if font_file and font_file != "Arial":
                    drawtext_params.append(f"fontfile='{font_file}'")
                    logger.info(f"使用字体文件: {font_file}")

                # 背景颜色设置
                if background_color and background_color.lower() not in [
                    "transparent",
                    "none",
                ]:
                    # 背景颜色格式与字体颜色完全一致
                    # 支持: #FFFFFF, white, 0xFFFFFF, black@0.5 等
                    drawtext_params.append("box=1")
                    drawtext_params.append(f"boxcolor={background_color}")
                    # 添加背景边距，让文字更清晰
                    drawtext_params.append("boxborderw=5")

                # 边框设置 - 增强边框效果
                if add_border:
                    drawtext_params.extend(
                        ["borderw=3", "bordercolor=black"]  # 增加边框宽度从2到3
                    )

                # 添加阴影效果，提高可读性
                drawtext_params.extend(
                    ["shadowx=2", "shadowy=2", "shadowcolor=black@0.5"]
                )

                filter_str = "drawtext=" + ":".join(drawtext_params)
                drawtext_filters.append(filter_str)

            # 组合所有滤镜
            if drawtext_filters:
                vf = ",".join(drawtext_filters)
            else:
                raise ValueError("没有生成任何字幕滤镜")

            # 构建FFmpeg命令
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                vf,
                "-c:a",
                "copy",  # 复制音频流，不重新编码
                "-c:v",
                "libx264",  # 视频编码器
                "-preset",
                "medium",  # 编码预设
                "-crf",
                "23",  # 质量设置
                output_path,
            ]

            logger.info(f"执行FFmpeg命令: {' '.join(cmd[:10])}...")  # 只显示前几个参数
            logger.info(
                f"字体大小: {font_size}, 位置: {position}, 字体文件: {font_file}"
            )

            if progress_callback:
                progress_callback(70, "正在生成硬字幕视频...")

            # 执行命令
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # 等待处理完成
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg处理失败: {stderr}")
                raise RuntimeError(f"视频处理失败: {stderr}")

            if progress_callback:
                progress_callback(100, "硬字幕视频生成完成")

            logger.info(f"硬字幕视频生成完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"生成硬字幕视频失败: {e}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise

    def add_subtitle_to_video(
        self,
        video_path: str,
        subtitle_text: str,
        font_name: str = "Arial",
        font_size: int = 24,
        font_color: str = "white",
        background_color: str = "transparent",
        add_border: bool = True,
        position: str = "bottom",
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> str:
        """
        为视频添加智能对齐的硬字幕

        Args:
            video_path: 视频文件路径
            subtitle_text: 字幕文本内容（纯文本，将与语音自动对齐）
            font_name: 字体名称
                支持: Arial, Times, Helvetica, DejaVu Sans, 微软雅黑 等
                注意: 中文字体名需要系统支持，建议使用英文字体名
            font_size: 字体大小 (8-128，推荐18-48)
            font_color: 字体颜色
                支持格式:
                - 英文颜色名: white, black, red, green, blue, yellow, cyan, magenta
                - 十六进制: #FFFFFF, #FF0000, #00FF00, #0000FF
                - RGB值: 0xFFFFFF, 0xFF0000
                - 透明度格式: white@0.8, #FF0000@0.5 (透明度0.0-1.0)
            background_color: 背景颜色
                支持格式完全相同于font_color:
                - 英文颜色名: white, black, red, green, blue, yellow, cyan, magenta
                - 十六进制: #FFFFFF, #FF0000, #00FF00, #0000FF
                - RGB值: 0xFFFFFF, 0xFF0000
                - 透明度格式: black@0.5, #000000@0.3 (推荐0.3-0.7)
                - 特殊值: transparent (透明背景), none (无背景)
            add_border: 是否添加边框 (提高可读性)
            position: 字幕位置
                支持: bottom(底部), top(顶部), middle(中央), left(左侧), right(右侧)
            progress_callback: 进度回调函数

        Returns:
            处理后的视频文件路径
        """
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        audio_path = None

        try:
            if progress_callback:
                progress_callback(5, "开始智能字幕处理...")

            # 步骤1: 从视频中提取音频
            if progress_callback:
                progress_callback(10, "从视频中提取音频...")

            audio_path = self._extract_audio_from_video(video_path)

            # 步骤2: 使用Whisper进行语音识别
            if progress_callback:
                progress_callback(20, "使用Whisper进行语音识别...")

            recognition_results = self._transcribe_audio_with_whisper(audio_path)

            # 步骤3: 将用户文本与识别结果对齐
            if progress_callback:
                progress_callback(50, "进行文本对齐...")

            aligned_segments = self._align_text_with_recognition(
                subtitle_text, recognition_results
            )

            if not aligned_segments:
                raise ValueError("无法生成字幕片段，请检查输入文本")

            # 步骤4: 生成硬字幕视频
            if progress_callback:
                progress_callback(60, "生成硬字幕视频...")

            output_path = self._create_hard_subtitle_video(
                video_path=video_path,
                subtitle_segments=aligned_segments,
                font_name=font_name,
                font_size=font_size,
                font_color=font_color,
                background_color=background_color,
                add_border=add_border,
                position=position,
                progress_callback=progress_callback,
            )

            logger.info(f"智能硬字幕处理完成: {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"添加智能字幕时发生错误: {e}")
            raise
        finally:
            # 清理临时文件
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                    logger.info(f"已清理临时音频文件: {audio_path}")
                except:
                    pass
