#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from typing import List, Tuple, Dict, Optional


def parse_subtitle_file(subtitle_path: str) -> List[str]:
    """
    解析简单的字幕文件
    """
    with open(subtitle_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def calculate_timings(
    subtitle_lines: List[str], video_duration: float
) -> List[Tuple[float, float, str]]:
    """
    根据视频长度自动计算每行字幕的时间
    返回格式: [(开始时间, 结束时间, 字幕文本), ...]
    """
    line_count = len(subtitle_lines)
    if line_count == 0:
        return []

    # 每行字幕平均时间
    duration_per_line = video_duration / line_count

    # 生成时间戳
    timings = []
    for i, line in enumerate(subtitle_lines):
        start_time = i * duration_per_line
        end_time = (i + 1) * duration_per_line
        timings.append((start_time, end_time, line))

    return timings


def add_subtitles(
    video_path: str,
    subtitle_timings: List[Tuple[float, float, str]],
    output_path: str,
    font_size: int = 24,
    font_color: str = "white",
    position: str = "bottom",
    add_border: bool = True,
):
    """
    将字幕添加到视频中
    """
    video = VideoFileClip(video_path)

    subtitle_clips = []
    for start_time, end_time, text in subtitle_timings:
        # 创建文本剪辑
        txt_clip = TextClip(
            text, fontsize=font_size, color=font_color, font="Arial-Unicode-MS"
        )

        # 设置文本位置
        if position == "bottom":
            txt_clip = txt_clip.set_position(("center", "bottom"))
        elif position == "top":
            txt_clip = txt_clip.set_position(("center", "top"))
        else:  # center
            txt_clip = txt_clip.set_position("center")

        # 添加黑色边框以提高可见度
        if add_border:
            txt_clip = TextClip(
                text,
                fontsize=font_size,
                color=font_color,
                font="Arial-Unicode-MS",
                stroke_color="black",
                stroke_width=2,
            )
            txt_clip = txt_clip.set_position(("center", "bottom"))

        # 设置时间范围
        txt_clip = txt_clip.set_start(start_time).set_end(end_time)
        subtitle_clips.append(txt_clip)

    # 合成最终视频
    final_video = CompositeVideoClip([video] + subtitle_clips)

    # 导出视频
    final_video.write_videofile(output_path, codec="libx264")

    # 释放资源
    video.close()
    final_video.close()


def main():
    parser = argparse.ArgumentParser(description="为视频添加字幕")
    parser.add_argument("--video_path", type=str, required=True, help="视频文件路径")
    parser.add_argument("--subtitle_path", type=str, required=True, help="字幕文件路径")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--font_size", type=int, default=24, help="字体大小")
    parser.add_argument("--font_color", type=str, default="white", help="字体颜色")
    parser.add_argument(
        "--position",
        type=str,
        default="bottom",
        choices=["top", "center", "bottom"],
        help="字幕位置",
    )
    parser.add_argument("--add_border", action="store_true", help="添加边框")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取输出文件路径
    video_filename = os.path.basename(args.video_path)
    output_filename = f"subtitled_{video_filename}"
    output_path = os.path.join(args.output_dir, output_filename)

    # 解析字幕文件
    subtitle_lines = parse_subtitle_file(args.subtitle_path)

    # 获取视频时长
    video = VideoFileClip(args.video_path)
    video_duration = video.duration
    video.close()

    # 计算时间戳
    subtitle_timings = calculate_timings(subtitle_lines, video_duration)

    # 添加字幕
    add_subtitles(
        args.video_path,
        subtitle_timings,
        output_path,
        font_size=args.font_size,
        font_color=args.font_color,
        position=args.position,
        add_border=args.add_border,
    )

    print(f"字幕添加完成，输出文件：{output_path}")


if __name__ == "__main__":
    main()
