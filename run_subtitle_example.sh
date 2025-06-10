#!/bin/bash

# 确保脚本在出错时停止执行
set -e

echo "字幕添加示例脚本"
echo "-------------------"

# 检查视频文件是否存在
VIDEO_FILE="example_video.mp4"
if [ ! -f "$VIDEO_FILE" ]; then
    echo "错误: 视频文件 $VIDEO_FILE 不存在。"
    echo "请先准备一个视频文件用于测试，并命名为 example_video.mp4"
    exit 1
fi

# 检查字幕文件是否存在
SUBTITLE_FILE="example_subtitle.txt"
if [ ! -f "$SUBTITLE_FILE" ]; then
    echo "错误: 字幕文件 $SUBTITLE_FILE 不存在。"
    exit 1
fi

# 运行字幕添加脚本
echo "正在添加字幕到视频..."
python add_subtitle.py \
    --video "$VIDEO_FILE" \
    --subtitle "$SUBTITLE_FILE" \
    --output_dir "./output" \
    --font_size 24 \
    --font_color "white" \
    --position "bottom" \
    --add_border

echo "处理完成！输出文件保存在 ./output 目录中"
