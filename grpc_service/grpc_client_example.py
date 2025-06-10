#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spark TTS gRPC 客户端示例

支持的功能：
1. 音色克隆 (clone) - 使用源音频克隆声音特征
2. 音频合成 (tts) - 使用指定声音合成文本
3. 视频音频提取 (extract) - 从视频中提取音频
4. 智能硬字幕 (subtitle) - 为视频添加智能对齐的硬字幕

智能硬字幕功能特点：
- 🎯 自动语音识别：使用 Whisper 识别视频中的语音内容
- 🧠 智能文本对齐：将用户提供的准确文本与识别结果进行时间对齐
- 🎬 硬字幕生成：直接将字幕烧录到视频画面中，兼容所有播放器
- ⚡ 零配置使用：无需手动设置时间戳，只需提供纯文本

使用示例：
# 基本字幕添加
python grpc_client_example.py subtitle --video my_video.mp4 --text "这是要添加的字幕内容"

# 自定义样式
python grpc_client_example.py subtitle --video my_video.mp4 --text "自定义字幕" --font-size 28 --font-color yellow --position top

# 从文件读取字幕
python grpc_client_example.py subtitle --video my_video.mp4 --subtitle-file subtitle.txt

# 上传本地视频并添加字幕
python grpc_client_example.py subtitle --upload /path/to/local_video.mp4 --video local_video.mp4 --text "字幕内容"
"""

import os
import sys
import time
import argparse
import grpc
import shutil
import uuid

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入生成的gRPC代码
import voice_service_pb2
import voice_service_pb2_grpc


def upload_file_to_task_dir(local_file_path, task_id, server_files_dir="files"):
    """
    将本地文件上传到服务器的任务目录中

    Args:
        local_file_path: 本地文件路径
        task_id: 任务ID
        server_files_dir: 服务器文件目录

    Returns:
        任务目录中的文件名
    """
    if not os.path.exists(local_file_path):
        print(f"错误: 本地文件不存在: {local_file_path}")
        return None

    # 创建目标目录
    task_dir = os.path.join(server_files_dir, f"dir_{task_id}")
    os.makedirs(task_dir, exist_ok=True)

    # 复制文件
    filename = os.path.basename(local_file_path)
    target_path = os.path.join(task_dir, filename)
    shutil.copy2(local_file_path, target_path)

    print(f"文件已上传到: {target_path}")
    return filename


def run_clone_voice(
    audio_name,
    target_text=None,
    voice_speed=1.0,
    task_id=None,
    task_step=None,
    prompt_text=None,
    server_address="voice-service:50051",
):
    """
    运行音色克隆任务

    Args:
        audio_name: 源音频文件名(不需要完整路径)
        target_text: 目标文本，如果为None则使用识别的文本
        voice_speed: 语音速度
        task_id: 任务ID，如果为None则自动生成
        task_step: 任务步骤，默认为None，将使用"0"
        prompt_text: 提示文本（可选）
        server_address: gRPC服务器地址，默认为 'voice-service:50051'
    """
    # 创建gRPC通道
    channel = grpc.insecure_channel(server_address)

    try:
        # 创建stub
        stub = voice_service_pb2_grpc.VoiceServiceStub(channel)

        # 创建请求
        request = voice_service_pb2.CloneVoiceRequest(
            audio_name=audio_name, voice_speed=voice_speed
        )

        # 如果指定了目标文本，添加到请求中
        if target_text:
            request.target_text = target_text

        # 如果指定了提示文本，添加到请求中
        if prompt_text:
            request.prompt_text = prompt_text

        # 如果指定了任务ID，添加到请求中
        if task_id:
            request.task_id = task_id

        # 如果指定了任务步骤，添加到请求中
        if task_step:
            request.task_step = task_step

        # 发送异步请求
        print(f"发送音色克隆请求:\n源音频: {audio_name}")
        if target_text:
            print(f"目标文本: {target_text}")
        else:
            print("目标文本: 使用识别结果")
        if prompt_text:
            print(f"提示文本: {prompt_text}")
        if task_id:
            print(f"任务ID: {task_id}")
        if task_step:
            print(f"任务步骤: {task_step}")

        # 发送请求，增加超时时间
        response = None
        try:
            response = stub.CloneVoice(
                request, timeout=300
            )  # 增加到300秒超时，适应200MB文件
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print("请求超时，但任务可能仍在处理中...")
                if task_id:
                    print(f"您可以稍后使用任务ID检查结果: {task_id}")
                    print(f"完整输出路径将是: /app/files/dir_{task_id}/[输出文件名]")
                return
            else:
                raise e

        # 如果响应为None（超时但没有任务ID），则退出
        if response is None:
            return

        # 打印任务ID和状态
        print(f"\n任务ID: {response.task_id}")
        print(f"状态: {response.status}")

        # 打印转录结果（如果有）
        if hasattr(response, "segments") and response.segments:
            print("\n源音频识别结果:")
            for segment in response.segments:
                print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")

        # 如果任务已完成并有输出文件名
        if (
            response.is_finished
            and response.status == "COMPLETED"
            and hasattr(response, "output_filename")
            and response.output_filename
        ):
            print(f"\n处理完成，输出文件: {response.output_filename}")
            print(
                f"完整输出路径: /app/files/dir_{response.task_id}/{response.output_filename}"
            )
            return

        # 由于我们不使用ProgressRequest，所以直接提示用户
        print("\n任务正在处理中，但我们无法获取实时进度。")
        print(f"完整输出路径将是: /app/files/dir_{response.task_id}/[输出文件名]")
        print(f"您可以稍后使用任务ID检查结果: {response.task_id}")

    except grpc.RpcError as e:
        print(f"RPC错误: {e.details()}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        channel.close()


def run_synthesize(
    text,
    voice_name=None,
    voice_speed=1.0,
    task_id=None,
    task_step=None,
    whisper_model=None,
    whisper_language=None,
    compute_type=None,
    prompt_text=None,
    server_address="voice-service:50051",
):
    """
    运行指定音色合成任务

    Args:
        text: 要合成的文本
        voice_name: 声音名称(可以是音频文件名)
        voice_speed: 语音速度
        task_id: 任务ID，如果为None则自动生成
        task_step: 任务步骤，默认为None，将使用"0"
        whisper_model: Whisper模型大小，可选 tiny, base, small, medium, large-v2
        whisper_language: 语音识别语言，默认为zh
        compute_type: 计算精度类型，可选 int8, float16, float32
        prompt_text: 提示文本（可选）
        server_address: gRPC服务器地址，默认为 'voice-service:50051'
    """
    # 创建gRPC通道
    channel = grpc.insecure_channel(server_address)

    try:
        # 创建stub
        stub = voice_service_pb2_grpc.VoiceServiceStub(channel)

        # 创建请求
        request = voice_service_pb2.SynthesizeRequest(
            text=text, voice_speed=voice_speed
        )

        # 如果指定了声音名称，添加到请求中
        if voice_name:
            request.voice_name = voice_name

        # 如果指定了任务ID，添加到请求中
        if task_id:
            request.task_id = task_id

        # 如果指定了任务步骤，添加到请求中
        if task_step:
            request.task_step = task_step

        # 如果指定了Whisper模型，添加到请求中
        if whisper_model:
            request.whisper_model = whisper_model

        # 如果指定了语言，添加到请求中
        if whisper_language:
            request.whisper_language = whisper_language

        # 如果指定了计算类型，添加到请求中
        if compute_type:
            request.compute_type = compute_type

        # 如果指定了提示文本，添加到请求中
        if prompt_text:
            request.prompt_text = prompt_text

        # 发送异步请求
        print(f"发送音频合成请求:\n文本: {text}")
        if voice_name:
            print(f"声音名称: {voice_name}")
        print(f"语音速度: {voice_speed}")
        if whisper_model:
            print(f"Whisper模型: {whisper_model}")
        if whisper_language:
            print(f"识别语言: {whisper_language}")
        if compute_type:
            print(f"计算精度: {compute_type}")
        if prompt_text:
            print(f"提示文本: {prompt_text}")
        if task_id:
            print(f"任务ID: {task_id}")
        if task_step:
            print(f"任务步骤: {task_step}")

        # 发送合成请求，增加超时时间
        response = None
        try:
            response = stub.Synthesize(
                request, timeout=300
            )  # 增加到300秒超时，适应200MB文件
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print("请求超时，但任务可能仍在处理中...")
                if task_id:
                    print(f"您可以稍后使用任务ID检查结果: {task_id}")
                    print(f"完整输出路径将是: /app/files/dir_{task_id}/[输出文件名]")
                return
            else:
                raise e

        # 如果响应为None（超时但没有任务ID），则退出
        if response is None:
            return

        # 打印任务ID和状态
        print(f"\n任务ID: {response.task_id}")
        print(f"状态: {response.status}")

        # 如果任务已完成并有输出文件名
        if (
            response.is_finished
            and response.status == "COMPLETED"
            and hasattr(response, "output_filename")
            and response.output_filename
        ):
            print(f"\n处理完成，输出文件: {response.output_filename}")
            print(
                f"完整输出路径: /app/files/dir_{response.task_id}/{response.output_filename}"
            )
            return

        # 由于我们不使用ProgressRequest，所以直接提示用户
        print("\n任务正在处理中，但我们无法获取实时进度。")
        print(f"完整输出路径将是: /app/files/dir_{response.task_id}/[输出文件名]")
        print(f"您可以稍后使用任务ID检查结果: {response.task_id}")

    except grpc.RpcError as e:
        print(f"RPC错误: {e.details()}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        channel.close()


def run_extract_audio(
    video_name,
    sample_rate=44100,
    mono=True,
    start_time=0.0,
    duration=10.0,
    auto_detect_voice=False,
    max_silence=2.0,
    task_id=None,
    task_step=None,
    upload_file=None,
    server_address="voice-service:50051",
):
    """
    从视频中提取音频（同步方式）

    Args:
        video_name: 视频文件名(不需要完整路径)
        sample_rate: 采样率，默认44100
        mono: 是否单声道，默认True
        start_time: 开始时间（秒），默认为0
        duration: 持续时间（秒），默认为10秒，设为0表示提取到结尾
        auto_detect_voice: 是否自动检测有声部分，默认False
        max_silence: 最大静音时长（秒），用于自动检测，默认2.0
        task_id: 任务ID，如果为None则自动生成
        task_step: 任务步骤，默认为None，将使用"0"
        upload_file: 要上传的本地文件路径，默认为None
        server_address: gRPC服务器地址，默认为 'voice-service:50051'
    """
    # 如果没有指定任务ID，生成一个
    if not task_id:
        task_id = str(uuid.uuid4())

    # 如果提供了要上传的文件
    if upload_file:
        # 检查文件是否存在
        if not os.path.exists(upload_file):
            print(f"错误: 本地文件不存在: {upload_file}")
            return

        # 上传文件
        uploaded_filename = upload_file_to_task_dir(upload_file, task_id)
        if uploaded_filename:
            # 使用上传后的文件名
            video_name = uploaded_filename
            print(f"将使用上传的文件: {video_name}")

    # 创建gRPC通道
    channel = grpc.insecure_channel(server_address)

    try:
        # 创建stub
        stub = voice_service_pb2_grpc.VoiceServiceStub(channel)

        # 创建请求
        request = voice_service_pb2.ExtractAudioRequest(
            video_name=video_name,
            sample_rate=sample_rate,
            mono=mono,
            start_time=start_time,
            duration=duration,
            auto_detect_voice=auto_detect_voice,
            max_silence=max_silence,
        )

        # 设置任务ID和步骤
        request.task_id = task_id
        if task_step:
            request.task_step = task_step

        # 发送请求
        print(f"发送从视频提取音频请求:\n视频文件: {video_name}")
        print(f"采样率: {sample_rate}")
        print(f"单声道: {mono}")
        print(f"开始时间: {start_time}秒")
        print(f"持续时间: {duration}秒" if duration > 0 else "持续时间: 直到结尾")
        print(f"自动检测有声部分: {'是' if auto_detect_voice else '否'}")
        if auto_detect_voice:
            print(f"最大静音时长: {max_silence}秒")
        print(f"任务ID: {task_id}")
        if task_step:
            print(f"任务步骤: {task_step}")

        print("\n正在处理，请稍候...")

        # 直接发送同步请求并获取响应
        response = stub.ExtractAudioFromVideo(request)

        # 打印结果
        print(f"\n任务ID: {response.task_id}")
        print(f"状态: {response.status}")
        print(f"消息: {response.message}")

        if response.is_finished:
            if response.status == "COMPLETED":
                print(f"\n处理完成，输出文件: {response.output_filename}")
                # 打印自动识别的提示文本
                if hasattr(response, "prompt_text") and response.prompt_text:
                    print(f"\n自动识别的提示文本: {response.prompt_text}")
                    print(f"\n可以使用此提示文本进行语音合成，如:")
                    print(
                        f'python grpc_client_example.py tts --text "要合成的文本" --prompt "{response.prompt_text}"'
                    )
            else:
                print(f"\n处理失败: {response.message}")
        else:
            print("\n处理未完成，请稍后查询结果")

    except grpc.RpcError as e:
        print(f"RPC错误: {e.details()}")
    finally:
        channel.close()


def run_add_subtitle(
    video_name,
    subtitle_text,
    task_id=None,
    task_step=None,
    font_name="Arial",
    font_size=24,
    font_color="white",
    add_border=True,
    position="bottom",
    upload_file=None,
    subtitle_file=None,
    server_address="voice-service:50051",
):
    """
    为视频添加智能对齐的硬字幕（同步方式）

    Args:
        video_name: 视频文件名(不需要完整路径)
        subtitle_text: 字幕文本内容（纯文本，将自动与语音对齐）
        task_id: 任务ID，如果为None则自动生成
        task_step: 任务步骤，默认为None，将使用"0"
        font_name: 字体名称，默认Arial
        font_size: 字体大小，默认24
        font_color: 字体颜色，默认white
        add_border: 是否添加边框，默认True
        position: 字幕位置，可选: bottom, top, middle
        upload_file: 要上传的本地视频文件路径，默认为None
        subtitle_file: 要上传的字幕文件，如果提供则使用文件内容替代subtitle_text
        server_address: gRPC服务器地址，默认为 'voice-service:50051'
    """
    # 如果没有指定任务ID，生成一个
    if not task_id:
        task_id = str(uuid.uuid4())

    # 如果提供了要上传的视频文件
    if upload_file:
        # 检查文件是否存在
        if not os.path.exists(upload_file):
            print(f"错误: 本地视频文件不存在: {upload_file}")
            return

        # 上传文件
        uploaded_filename = upload_file_to_task_dir(upload_file, task_id)
        if uploaded_filename:
            # 使用上传后的文件名
            video_name = uploaded_filename
            print(f"将使用上传的视频文件: {video_name}")

    # 如果提供了字幕文件，读取内容
    if subtitle_file:
        if not os.path.exists(subtitle_file):
            print(f"错误: 本地字幕文件不存在: {subtitle_file}")
            return

        # 读取字幕文件内容
        try:
            with open(subtitle_file, "r", encoding="utf-8") as f:
                subtitle_text = f.read().strip()
            print(f"已读取字幕文件: {subtitle_file}")
            print(
                f"字幕内容预览: {subtitle_text[:100]}{'...' if len(subtitle_text) > 100 else ''}"
            )
        except Exception as e:
            print(f"读取字幕文件失败: {e}")
            return

    # 检查字幕文本
    if not subtitle_text or not subtitle_text.strip():
        print("错误: 字幕文本不能为空")
        return

    # 创建gRPC通道
    channel = grpc.insecure_channel(server_address)

    try:
        # 创建stub
        stub = voice_service_pb2_grpc.VoiceServiceStub(channel)

        # 创建请求
        request = voice_service_pb2.AddSubtitleRequest(
            video_name=video_name,
            subtitle_text=subtitle_text,
            font_name=font_name,
            font_size=font_size,
            font_color=font_color,
            add_border=add_border,
            position=position,
        )

        # 设置任务ID和步骤
        request.task_id = task_id
        if task_step:
            request.task_step = task_step

        # 发送请求
        print(f"\n🎬 发送智能硬字幕添加请求:")
        print(f"📹 视频文件: {video_name}")
        print(f"📝 字幕文本长度: {len(subtitle_text)}字符")
        print(f"🎨 字体样式: {font_name}, 大小: {font_size}, 颜色: {font_color}")
        print(f"🖼️  添加边框: {'是' if add_border else '否'}")
        print(f"📍 位置: {position}")
        print(f"🆔 任务ID: {task_id}")
        if task_step:
            print(f"📊 任务步骤: {task_step}")

        print(f"\n📄 字幕内容预览:")
        preview_text = (
            subtitle_text[:200] + "..." if len(subtitle_text) > 200 else subtitle_text
        )
        print(f"   {preview_text}")

        print(f"\n⚡ 智能处理流程:")
        print(f"   1. 从视频中提取音频")
        print(f"   2. 使用Whisper进行语音识别")
        print(f"   3. 智能对齐用户文本与识别结果")
        print(f"   4. 生成硬字幕视频")
        print(f"\n⏳ 正在处理，请稍候...")

        # 发送同步请求并获取响应，增加超时时间
        try:
            response = stub.AddSubtitleToVideo(request, timeout=600)  # 10分钟超时
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print("⚠️  请求超时，但任务可能仍在处理中...")
                print(f"📁 您可以稍后检查输出目录: /app/files/dir_{task_id}/")
                return
            else:
                raise e

        # 打印结果
        print(f"\n📊 处理结果:")
        print(f"🆔 任务ID: {response.task_id}")
        print(f"📈 状态: {response.status}")
        print(f"💬 消息: {response.message}")

        if response.is_finished:
            if response.status == "COMPLETED":
                print(f"\n✅ 处理完成!")
                print(f"📁 输出文件: {response.output_filename}")
                print(
                    f"📂 完整路径: /app/files/dir_{response.task_id}/{response.output_filename}"
                )
                print(f"\n🎉 智能硬字幕视频生成成功!")
                print(f"💡 提示: 硬字幕已直接烧录到视频中，兼容所有播放器")
            else:
                print(f"\n❌ 处理失败: {response.message}")
        else:
            print(f"\n⏳ 处理未完成，请稍后查询结果")
            print(f"📁 输出目录: /app/files/dir_{response.task_id}/")

    except grpc.RpcError as e:
        print(f"❌ RPC错误: {e.details()}")
        print(f"💡 提示: 请检查服务器是否正常运行，以及视频文件是否存在")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
    finally:
        channel.close()


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="声音合成gRPC客户端示例")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 通用参数
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--task-id", help="任务ID，如果不指定则自动生成")
    parent_parser.add_argument("--task-step", default="0", help="任务步骤，默认为0")
    parent_parser.add_argument(
        "--server",
        "-s",
        default="localhost:50051",
        help="服务器地址，默认localhost:50051",
    )

    # 克隆音色命令
    clone_parser = subparsers.add_parser(
        "clone", help="克隆音色", parents=[parent_parser]
    )
    clone_parser.add_argument("--audio", "-a", required=True, help="源音频文件名")
    clone_parser.add_argument(
        "--text", "-t", help="目标文本，如果不指定则使用识别出的文本"
    )
    clone_parser.add_argument(
        "--speed", "-p", type=float, default=1.0, help="语音速度，默认1.0"
    )
    clone_parser.add_argument("--prompt", "-pt", help="提示文本，可选")

    # 合成音频命令
    tts_parser = subparsers.add_parser("tts", help="合成音频", parents=[parent_parser])
    tts_parser.add_argument("--text", "-t", required=True, help="要合成的文本")
    tts_parser.add_argument("--voice", "-v", help="声音名称，可选")
    tts_parser.add_argument(
        "--speed", "-p", type=float, default=1.0, help="语音速度，默认1.0"
    )
    tts_parser.add_argument("--prompt", "-pt", help="提示文本，可选")
    tts_parser.add_argument(
        "--whisper-model",
        "-w",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Whisper模型大小，可选 tiny, base, small, medium, large-v2，默认为medium",
    )
    tts_parser.add_argument(
        "--whisper-language",
        "-l",
        choices=["zh", "en", "ja", "ko"],
        help="语音识别语言，默认为zh",
    )
    tts_parser.add_argument(
        "--compute-type",
        "-c",
        choices=["int8", "float16", "float32"],
        help="计算精度类型，可选 int8, float16, float32，默认为float16",
    )

    # 从视频提取音频命令
    extract_parser = subparsers.add_parser(
        "extract", help="从视频提取音频", parents=[parent_parser]
    )
    extract_parser.add_argument("--video", "-v", required=True, help="视频文件名")
    extract_parser.add_argument(
        "--sample-rate", "-r", type=int, default=44100, help="采样率，默认44100"
    )
    extract_parser.add_argument(
        "--mono", "-m", action="store_true", default=True, help="是否单声道，默认是"
    )
    extract_parser.add_argument(
        "--start", "-st", type=float, default=0.0, help="开始时间（秒），默认0.0"
    )
    extract_parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="持续时间（秒），默认10.0，设为0表示提取到结尾",
    )
    extract_parser.add_argument(
        "--auto-detect",
        "-a",
        action="store_true",
        default=False,
        help="是否自动检测有声部分，默认否",
    )
    extract_parser.add_argument(
        "--max-silence",
        type=float,
        default=2.0,
        help="最大静音时长（秒），用于自动检测，默认2.0",
    )
    extract_parser.add_argument(
        "--upload",
        "-u",
        help="上传本地视频文件到服务器（提供本地文件路径）",
    )
    extract_parser.add_argument(
        "--auto-prompt",
        action="store_true",
        default=True,
        help="自动识别音频内容作为提示文本，默认是",
    )

    # 为视频添加字幕命令
    subtitle_parser = subparsers.add_parser(
        "subtitle", help="为视频添加智能对齐的硬字幕", parents=[parent_parser]
    )
    subtitle_parser.add_argument("--video", "-v", required=True, help="视频文件名")
    subtitle_parser.add_argument(
        "--text",
        "-t",
        help="字幕文本内容（纯文本，将自动与视频语音对齐），如果不提供则必须通过--subtitle-file指定字幕文件",
    )
    subtitle_parser.add_argument(
        "--subtitle-file", "-sf", help="字幕文件路径，包含纯文本内容（不需要时间戳）"
    )
    subtitle_parser.add_argument(
        "--font", "-f", default="Arial", help="字体名称，默认Arial"
    )
    subtitle_parser.add_argument(
        "--font-size", "-fs", type=int, default=24, help="字体大小，默认24"
    )
    subtitle_parser.add_argument(
        "--font-color", "-fc", default="white", help="字体颜色，默认white"
    )
    subtitle_parser.add_argument(
        "--no-border", "-nb", action="store_true", help="不添加边框（默认添加）"
    )
    subtitle_parser.add_argument(
        "--position",
        "-p",
        default="bottom",
        choices=["bottom", "top", "middle"],
        help="字幕位置，可选: bottom, top, middle，默认bottom",
    )
    subtitle_parser.add_argument(
        "--upload", "-u", help="上传本地视频文件到服务器（提供本地文件路径）"
    )

    args = parser.parse_args()

    if args.command == "clone":
        run_clone_voice(
            audio_name=args.audio,
            target_text=args.text,
            voice_speed=args.speed,
            task_id=args.task_id,
            task_step=args.task_step,
            prompt_text=args.prompt,
            server_address=args.server,
        )
    elif args.command == "tts":
        run_synthesize(
            text=args.text,
            voice_name=args.voice,
            voice_speed=args.speed,
            task_id=args.task_id,
            task_step=args.task_step,
            whisper_model=args.whisper_model,
            whisper_language=args.whisper_language,
            compute_type=args.compute_type,
            prompt_text=args.prompt,
            server_address=args.server,
        )
    elif args.command == "extract":
        run_extract_audio(
            video_name=args.video,
            sample_rate=args.sample_rate,
            mono=args.mono,
            start_time=args.start,
            duration=args.duration,
            auto_detect_voice=args.auto_detect,
            max_silence=args.max_silence,
            task_id=args.task_id,
            task_step=args.task_step,
            upload_file=args.upload,
            server_address=args.server,
        )
    elif args.command == "subtitle":
        # 检查必要参数
        if not args.text and not args.subtitle_file:
            print("错误: 必须提供字幕文本(--text)或字幕文件(--subtitle-file)")
            sys.exit(1)

        run_add_subtitle(
            video_name=args.video,
            subtitle_text=args.text if args.text else "",
            task_id=args.task_id,
            task_step=args.task_step,
            font_name=args.font,
            font_size=args.font_size,
            font_color=args.font_color,
            add_border=not args.no_border,
            position=args.position,
            upload_file=args.upload,
            subtitle_file=args.subtitle_file,
            server_address=args.server,
        )
    else:
        parser.print_help()
