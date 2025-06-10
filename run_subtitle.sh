#!/bin/bash

# 检查必需参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <视频路径> <字幕文件路径> [输出目录]"
    echo "例如: $0 ./video.mp4 ./subtitle.txt ./output"
    exit 1
fi

VIDEO_PATH="$1"
SUBTITLE_PATH="$2"
OUTPUT_DIR="${3:-output}" # 如果没有提供输出目录，则默认为"output"

# 检查文件是否存在
if [ ! -f "$VIDEO_PATH" ]; then
    echo "错误: 视频文件 '$VIDEO_PATH' 不存在"
    exit 1
fi

if [ ! -f "$SUBTITLE_PATH" ]; then
    echo "错误: 字幕文件 '$SUBTITLE_PATH' 不存在"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 设置字幕参数
FONT_SIZE=24
FONT_COLOR="white"
POSITION="bottom"
ADD_BORDER="--add_border" # 默认添加边框

echo "开始处理视频: $VIDEO_PATH"
echo "使用字幕文件: $SUBTITLE_PATH"
echo "输出目录: $OUTPUT_DIR"

# 运行字幕添加程序
python add_subtitle.py \
    --video_path "$VIDEO_PATH" \
    --subtitle_path "$SUBTITLE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --font_size "$FONT_SIZE" \
    --font_color "$FONT_COLOR" \
    --position "$POSITION" \
    $ADD_BORDER

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "处理完成！"
else
    echo "处理过程中出错，请检查日志。"
    exit 1
fi
