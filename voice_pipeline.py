import os
import sys
import gc
import torch
import time
import numpy as np
import logging
import uuid
import tempfile
from typing import Optional, Dict, Tuple, List, Any, Union
from datetime import datetime
from tqdm import tqdm
import whisper
import soundfile as sf

# 检查CUDA是否可用
if not torch.cuda.is_available():
    print("错误：未检测到CUDA，程序终止。")
    sys.exit(1)

# 设置CUDA环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.SparkTTS import SparkTTS
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("voice_processor.log")],
)
logger = logging.getLogger("voice_processor")


class VoiceProcessor:
    """语音处理器，用于语音合成和识别"""

    def __init__(
        self,
        model_dir: str = "pretrained_models/Spark-TTS-0.5B",
        whisper_model_size: str = "medium",
        compute_type: str = "float16",
        language: str = "zh",
        output_dir: str = "outputs",
        max_new_tokens: int = 3000,  # 增加 max_new_tokens 参数
    ):
        """
        初始化语音处理器

        Args:
            model_dir: Spark-TTS模型路径
            whisper_model_size: Whisper模型大小，可选 tiny/base/small/medium/large-v2
            compute_type: 计算类型，可选 int8/float16/float32
            language: 语言，可选 zh/en
            output_dir: 输出目录
            max_new_tokens: TTS模型生成的最大token数量，默认3000
        """
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到CUDA，程序终止。")

        self.model_dir = model_dir
        self.whisper_model_size = whisper_model_size
        self.compute_type = compute_type
        self.language = language
        self.output_dir = output_dir
        self.device = "cuda"  # 强制使用CUDA
        self.max_new_tokens = max_new_tokens  # 保存 max_new_tokens 参数

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"正在初始化语音处理器，使用设备: {self.device}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"当前GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"最大生成token数: {self.max_new_tokens}")

        # 懒加载模型
        self._tts_model = None
        self._whisper_model = None

        logger.info("语音处理器初始化完成")

    def _load_whisper_model(self):
        """加载Whisper模型"""
        try:
            if self._whisper_model is None:
                logger.info(f"正在加载Whisper模型，大小: {self.whisper_model_size}")
                logger.info(f"使用设备: {self.device}")

                # 设置国内镜像源
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

                # 尝试直接使用官方模型名称加载
                try:
                    logger.info(f"尝试使用官方模型名称加载: {self.whisper_model_size}")
                    self._whisper_model = whisper.load_model(
                        self.whisper_model_size,  # 使用模型名称 "small", "medium" 等
                        device=self.device,
                        download_root=os.path.expanduser("/app/whisper_model"),
                    )
                    logger.info("Whisper模型加载成功")
                    return True
                except Exception as e:
                    logger.error(f"使用官方模型名称加载失败: {e}")

                # 尝试查找本地 .pt 文件
                model_file = os.path.join(
                    os.path.expanduser("/app/whisper_model"),
                    f"{self.whisper_model_size}.pt",
                )
                if os.path.isfile(model_file):
                    logger.info(f"发现本地模型文件: {model_file}")
                    try:
                        self._whisper_model = whisper.load_model(
                            model_file, device=self.device  # 使用 .pt 文件路径
                        )
                        logger.info("Whisper模型加载成功")
                        return True
                    except Exception as e:
                        logger.error(f"加载本地模型文件失败: {e}")
                else:
                    logger.warning(f"未找到本地模型文件: {model_file}")

                # 从镜像下载模型（通过使用官方模型名称）
                logger.info("尝试从镜像下载模型...")
                try:
                    self._whisper_model = whisper.load_model(
                        self.whisper_model_size,
                        device=self.device,
                        download_root=os.path.expanduser("/app/whisper_model"),
                    )
                    logger.info("Whisper模型下载并加载成功")
                    return True
                except Exception as e:
                    logger.error(f"从镜像下载模型失败: {e}")
                    return False
            return True
        except Exception as e:
            logger.error(f"加载Whisper模型失败: {e}")
            return False

    def _load_tts_model(self):
        """加载TTS模型"""
        try:
            if self._tts_model is None:
                from cli.SparkTTS import SparkTTS

                logger.info(f"正在加载Spark-TTS模型，路径: {self.model_dir}")
                self._tts_model = SparkTTS(
                    model_dir=self.model_dir,
                    device=self.device,
                )
                logger.info("Spark-TTS模型加载完成")
            return True
        except Exception as e:
            logger.error(f"加载Spark-TTS模型失败: {e}")
            return False

    def _clean_gpu_memory(self):
        """清理GPU内存"""
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def transcribe(self, audio_path: str) -> List[Dict]:
        """
        使用Whisper识别音频内容

        Args:
            audio_path: 音频文件路径

        Returns:
            包含识别结果的字典列表，每个字典包含文本、开始时间和结束时间
        """
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return []

        if not self._load_whisper_model():
            logger.error("Whisper模型加载失败，无法进行语音识别")
            return []

        try:
            logger.info(f"开始识别音频: {audio_path}")
            # 使用原始 whisper 的转录方法
            result = self._whisper_model.transcribe(
                audio_path, language=self.language, task="transcribe"
            )

            segments = []
            for segment in result["segments"]:
                segments.append(
                    {
                        "text": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"],
                    }
                )

            logger.info(f"音频识别完成，共 {len(segments)} 个片段")
            return segments

        except Exception as e:
            logger.exception(f"识别音频时发生错误: {e}")
            return []

        finally:
            self._clean_gpu_memory()

    def _split_text(self, text: str, max_length: int = 50) -> List[str]:
        """
        将文本分割成合适的段落。
        此方法保留供参考，但在当前实现中不再主动使用。

        Args:
            text: 要分割的文本
            max_length: 段落的最大长度

        Returns:
            分割后的文本段落列表
        """
        if not text:
            return []

        # 如果文本长度小于最大长度，直接返回
        if len(text) <= max_length:
            return [text]

        # 简单地按句子分割
        segments = []
        current_segment = ""

        # 句子结束标记
        sentence_ends = [".", "。", "!", "！", "?", "？", ";", "；", "\n"]

        for char in text:
            current_segment += char

            # 如果当前段落长度达到最大长度或者遇到句子结束标记
            if len(current_segment) >= max_length or char in sentence_ends:
                segments.append(current_segment)
                current_segment = ""

        # 添加最后一个段落（如果有）
        if current_segment:
            segments.append(current_segment)

        return segments

    def _crop_audio_if_needed(
        self, audio_path: str, target_duration: float = None
    ) -> str:
        """
        如果音频长度明显长于目标长度，则裁剪音频。适用于模型因音频时长不匹配导致合成问题的情况。

        Args:
            audio_path: 源音频文件路径
            target_duration: 目标持续时间（秒）。如果为None，则只提取前10秒

        Returns:
            处理后的音频路径（可能是新文件或原始文件）
        """
        if not target_duration:
            target_duration = 10.0  # 默认取前10秒

        try:
            # 读取音频文件
            data, sample_rate = sf.read(audio_path)
            duration = len(data) / sample_rate

            # 如果音频时长小于目标时长的1.5倍，不需要裁剪
            if duration <= target_duration * 1.5:
                logger.info(f"音频长度({duration:.1f}秒)适中，不需要裁剪")
                return audio_path

            logger.info(
                f"音频长度({duration:.1f}秒)明显长于目标长度({target_duration:.1f}秒)，进行裁剪"
            )

            # 计算要裁剪的样本数
            samples_to_keep = int(target_duration * sample_rate)

            # 选择音频的前一部分（可以是开头，也可以选择有声音的部分）
            cropped_data = data[:samples_to_keep]

            # 创建临时文件保存裁剪后的音频
            temp_dir = os.path.join(self.output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            cropped_path = os.path.join(
                temp_dir, f"cropped_{os.path.basename(audio_path)}"
            )
            sf.write(cropped_path, cropped_data, sample_rate)

            logger.info(
                f"音频裁剪成功，保存到: {cropped_path}，新长度: {target_duration:.1f}秒"
            )
            return cropped_path

        except Exception as e:
            logger.error(f"裁剪音频时发生错误: {e}")
            return audio_path  # 出错时返回原始音频路径

    def process_voice(
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
            voice_speed: 音频速度调整，可选
            prompt_text: 提示文本，可选
        Returns:
            元组，包含源音频识别结果和生成的音频文件路径
        """
        if not os.path.exists(source_audio):
            logger.error(f"源音频文件不存在: {source_audio}")
            return [], ""

        # 优先加载Whisper模型进行识别
        if not self._load_whisper_model():
            logger.error("Whisper模型加载失败，无法进行语音识别")
            return [], ""

        # 识别源音频特征
        try:
            # logger.info(f"开始识别源音频: {source_audio}")
            # recognition_results = self.transcribe(source_audio)

            # # 打印识别结果
            # logger.info("\n识别结果:")
            # for result in recognition_results:
            #     logger.info(
            #         f"[{result['start']:.1f}s -> {result['end']:.1f}s] {result['text']}"
            #     )

            # # 如果未提供提示文本，使用识别结果的所有片段拼接，“，”分隔
            # if len(prompt_text) == 0 and recognition_results:
            #     prompt_text = "，".join(
            #         [result["text"] for result in recognition_results]
            #     )
            #     logger.info(f"未提供提示文本，使用识别结果: {prompt_text}")

            # if not target_text:
            #     logger.error("未提供目标文本且识别结果为空")
            #     return recognition_results, ""

            # 根据目标文本长度，估计音频时长
            estimated_audio_duration = len(target_text) * 0.2  # 每字符约0.2秒

            # 裁剪源音频，使其与目标文本长度更匹配
            processed_audio = self._crop_audio_if_needed(
                source_audio,
                target_duration=min(
                    15.0, max(5.0, estimated_audio_duration)
                ),  # 最短5秒，最长15秒
            )
            logger.info(f"processed_audio:{processed_audio}")

            if processed_audio != source_audio:
                logger.info(f"使用裁剪后的音频作为提示: {processed_audio}")
                # 使用裁剪后的音频重新识别
                cropped_recognition_results = self.transcribe(processed_audio)
                if cropped_recognition_results:
                    logger.info(
                        f"裁剪后音频识别结果: {cropped_recognition_results[0]['text']}"
                    )
                    # 使用裁剪后的音频和识别结果
                    source_audio = processed_audio
                    recognition_results = cropped_recognition_results

        except Exception as e:
            logger.exception(f"识别源音频时发生错误: {e}")
            return [], ""

        # 加载TTS模型进行合成
        if not self._load_tts_model():
            logger.error("TTS模型加载失败，无法进行语音合成")
            return recognition_results, ""

        try:
            logger.info(f"开始合成音频，使用目标文本: {target_text}")
            logger.info(f"文本完整长度: {len(target_text)} 字符")

            # 创建输出文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"voice_clone_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
            output_path = os.path.join(self.output_dir, output_filename)

            # 根据文本长度动态调整max_new_tokens参数
            # 每个汉字大约需要10-15个token
            text_length = len(target_text)
            estimated_tokens = text_length * 15  # 估计所需token数量

            # 如果估计token数超过默认值，增加max_new_tokens
            max_new_tokens = max(self.max_new_tokens, estimated_tokens)
            logger.info(
                f"根据文本长度 {text_length} 字符，设置 max_new_tokens={max_new_tokens}"
            )

            # 直接使用完整文本合成音频，传递max_new_tokens参数
            try:
                logger.info("开始合成完整音频...")
                speed = "moderate"  # 默认中等语速
                if voice_speed is not None:
                    if voice_speed < 0.5:
                        speed = "very_low"
                    elif voice_speed < 0.8:
                        speed = "low"
                    elif voice_speed < 1.2:
                        speed = "moderate"
                    elif voice_speed < 1.5:
                        speed = "high"
                    else:
                        speed = "very_high"
                    logger.info(f"输入语速值 {voice_speed} 映射到速度级别: {speed}")

                # 检查提示音频长度与目标文本长度的匹配性
                # 不再使用prompt_text参数，只使用音频特征来捕获声音特点
                # 这样可以避免录制音频文本混入合成结果
                logger.info("不使用提示文本内容，只使用音频特征来提取声音特点")

                wav = self._tts_model.inference(
                    text=target_text,
                    prompt_speech_path=source_audio,
                    prompt_text=prompt_text,  # 可以选择是否提供提示文本
                    gender=None,
                    pitch="moderate",
                    speed=speed,
                    max_new_tokens=max_new_tokens,  # 使用计算后的max_new_tokens值
                )

                # 判断返回的wav是张量还是numpy数组
                if isinstance(wav, torch.Tensor):
                    wav_numpy = wav.cpu().numpy()
                else:
                    wav_numpy = wav

                logger.info(f"音频合成成功，长度: {len(wav_numpy)} 采样点")

                # 保存音频
                sf.write(output_path, wav_numpy, self._tts_model.sample_rate)
                logger.info(f"音频保存到: {output_path}")

                return recognition_results, output_path

            except Exception as e:
                logger.exception(f"合成音频时发生错误: {e}")

                # 如果出错，生成一个简单的音频作为备选
                logger.warning("生成一个简单的音频作为备选...")
                duration = len(target_text) * 0.2  # 每字符0.2秒
                t = np.linspace(
                    0, duration, int(self._tts_model.sample_rate * duration)
                )
                wav_numpy = np.sin(2 * np.pi * 440 * t)  # 基础音调

                fallback_filename = f"fallback_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
                fallback_path = os.path.join(self.output_dir, fallback_filename)
                sf.write(fallback_path, wav_numpy, self._tts_model.sample_rate)

                logger.info(f"备选音频保存到: {fallback_path}")
                return recognition_results, fallback_path

        except Exception as e:
            logger.exception(f"合成音频时发生错误: {e}")
            return recognition_results, ""

        finally:
            self._clean_gpu_memory()

    def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        voice_speed: Optional[float] = None,
        prompt_text: Optional[str] = None,
    ) -> str:
        """
        使用预设声音合成文本

        Args:
            text: 要合成的文本
            voice_name: 声音名称，可以是音频文件路径
            voice_speed: 语音速度，可选，1.0为正常速度。
                        会被映射到以下五个档位：
                        < 0.6: very_low
                        0.6-0.8: low
                        0.8-1.2: moderate
                        1.2-1.5: high
                        > 1.5: very_high
            prompt_text: 提示文本，如果为None且voice_name是音频文件，则自动识别

        Returns:
            生成的音频文件路径
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # 生成输出文件路径
        timestamp = int(time.time())
        output_path = os.path.join(self.output_dir, f"tts_{timestamp}.wav")

        if not self._load_tts_model():
            logger.error("TTS模型加载失败，无法合成音频")
            return ""

        try:
            # 将数值速度映射到字符串速度级别
            speed = "moderate"  # 默认中等语速
            if voice_speed is not None:
                if voice_speed < 0.5:
                    speed = "very_low"
                elif voice_speed < 0.8:
                    speed = "low"
                elif voice_speed < 1.2:
                    speed = "moderate"
                elif voice_speed < 1.5:
                    speed = "high"
                else:
                    speed = "very_high"
                logger.info(f"输入语速值 {voice_speed} 映射到速度级别: {speed}")

            logger.info(f"开始合成音频，使用文本: {text}")

            # 如果提供了声音名称且它是一个文件路径，使用它作为提示音频
            if voice_name:
                logger.info(f"提供了声音名称: {voice_name}")
                if os.path.isfile(voice_name):
                    logger.info(
                        f"声音名称是有效的文件路径，将使用提示音频: {voice_name}"
                    )

                    # 如果prompt_text为空，使用Whisper识别音频内容
                    if len(prompt_text) == 0:
                        logger.info("未提供prompt_text，尝试使用Whisper识别音频内容...")
                        if self._load_whisper_model():
                            recognition_results = self.transcribe(voice_name)
                            if recognition_results:
                                prompt_text = "，".join(
                                    [result["text"] for result in recognition_results]
                                )
                                logger.info(
                                    f"成功识别音频内容作为prompt_text: {prompt_text}"
                                )
                            else:
                                logger.warning("音频识别失败，将不使用prompt_text")
                        else:
                            logger.warning("Whisper模型加载失败，将不使用prompt_text")

                    # 使用提示音频时，不设置gender参数
                    wav = self._tts_model.inference(
                        text=text,
                        prompt_speech_path=voice_name,
                        prompt_text=prompt_text,  # 可以是None或识别出的文本
                        gender=None,
                        speed=speed,
                        pitch="moderate",
                    )
                else:
                    # 如果不是有效的音频文件，使用控制参数合成
                    logger.warning(
                        f"声音名称不是有效的文件路径: {voice_name}，将使用默认参数"
                    )
                    gender = "female"
                    pitch = "moderate"
                    logger.info(
                        f"使用默认参数合成: 性别={gender}, 音调={pitch}, 语速={speed}"
                    )
                    wav = self._tts_model.inference(
                        text=text,
                        gender=gender,
                        pitch=pitch,
                        speed=speed,
                        prompt_text=prompt_text,
                    )
            else:
                # 使用默认参数合成
                logger.info("未提供声音名称，使用默认参数")
                gender = "female"
                pitch = "moderate"
                logger.info(
                    f"使用默认参数合成: 性别={gender}, 音调={pitch}, 语速={speed}"
                )
                wav = self._tts_model.inference(
                    text=text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                    prompt_text=prompt_text,
                )

            # 保存波形到文件
            import soundfile as sf

            # 检查wav是张量还是numpy数组
            if isinstance(wav, torch.Tensor):
                wav_numpy = wav.cpu().numpy()
            else:
                wav_numpy = wav

            sf.write(output_path, wav_numpy, self._tts_model.sample_rate)

            logger.info(f"音频合成完成，保存到: {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"合成音频时发生错误: {e}")
            return ""

        finally:
            self._clean_gpu_memory()


# 模拟Spark TTS引擎
# class SparkTTS:
#     """模拟的Spark TTS引擎，实际项目中应替换为真实实现"""

#     def __init__(self, model_dir, device="cuda", compute_type="int8"):
#         self.model_dir = model_dir
#         self.device = device
#         self.compute_type = compute_type
#         self.sample_rate = 24000  # 设置采样率
#         logger.info(
#             f"初始化Spark TTS模型: {model_dir}, 设备: {device}, 计算类型: {compute_type}"
#         )

#     def inference(
#         self,
#         text,
#         prompt_speech_path=None,
#         prompt_text=None,
#         speed=1.0,  # 改为直接使用浮点数表示速度
#         gender=None,
#         pitch=None,
#     ):
#         """使用提示音频合成文本"""
#         logger.info(
#             f"合成音频: {text}, 提示音频: {prompt_speech_path}, 提示文本: {prompt_text}, 语速: {speed}倍"
#         )

#         # 根据文本长度动态生成对应长度的音频
#         # 估算：假设每个字符需要0.2秒
#         text_length = len(text)
#         duration = max(2.0, text_length * 0.2)  # 最少2秒，每个字符0.2秒

#         logger.info(f"根据文本长度 {text_length} 字符生成 {duration:.2f} 秒音频")

#         t = np.linspace(0, duration, int(self.sample_rate * duration))

#         # 直接使用speed值调整音频时长
#         if speed != 1.0:
#             duration = duration / speed
#             t = np.linspace(0, duration, int(self.sample_rate * duration))
#             logger.info(f"应用语速 {speed}倍，调整后音频长度: {duration:.2f} 秒")

#         # 生成更复杂的音频，不只是单一频率
#         wav = np.sin(2 * np.pi * 440 * t)  # 基础音调

#         # 添加一些变化以模拟真实语音
#         for i, char in enumerate(text):
#             # 根据字符位置计算时间位置
#             char_pos = int((i / text_length) * len(t))
#             if char_pos < len(t) - 100:  # 确保不越界
#                 # 在对应位置添加振幅变化
#                 wav[char_pos : char_pos + 100] *= 1.2

#         return torch.from_numpy(wav).float()

#     def synthesize_with_preset_voice(self, text, voice_name, output_path, speed=None):
#         """使用预设声音合成音频"""
#         logger.info(
#             f"使用预设声音合成音频: {voice_name}, 文本: {text}, 输出到: {output_path}"
#         )

#         # 调用inference方法以保持一致性
#         wav = self.inference(text=text, speed=speed)

#         # 保存音频
#         if isinstance(wav, torch.Tensor):
#             wav = wav.cpu().numpy()

#         sf.write(output_path, wav, self.sample_rate)
#         return output_path

#     def synthesize_with_default_voice(self, text, output_path, speed=None):
#         """使用默认声音合成音频"""
#         logger.info(f"使用默认声音合成音频: {text}, 输出到: {output_path}")

#         # 调用inference方法以保持一致性
#         wav = self.inference(text=text, speed=speed)

#         # 保存音频
#         if isinstance(wav, torch.Tensor):
#             wav = wav.cpu().numpy()

#         sf.write(output_path, wav, self.sample_rate)
#         return output_path


# 使用示例
if __name__ == "__main__":
    try:
        # 创建命令行参数解析器
        parser = argparse.ArgumentParser(description="语音处理工具")
        parser.add_argument(
            "--input1", default="./inputs/output_audio_123.wav", help="源音频文件路径"
        )
        parser.add_argument(
            "--input2",
            default="今天这条视频，新老朋友刷到了，给我全部收藏起来，仔细听我的叮嘱，接下来的天气，虽然是向阳而来，但是中途变幻莫测，稍有不慎就将望阳兴叹，先跟老朋友说几句，前两天我直播聊的两件衣服，一定焊死在你的衣柜里面，虽然已经有两个板子了，可以再穿一周，再说新朋友，现在天气虽然重要，但是不要本末倒置，没方向的朋友，晚上的两件新衣服",
            help="目标文本",
        )
        parser.add_argument("--output_dir", default="./outputs", help="输出目录路径")
        parser.add_argument(
            "--whisper_model_size", default="large-v2", help="Whisper模型大小"
        )
        parser.add_argument("--compute_type", default="float16", help="计算类型")
        parser.add_argument("--language", default="zh", help="语言")
        parser.add_argument("--voice_speed", type=float, default=1.0, help="语音速度")

        args = parser.parse_args()

        # 创建语音处理器实例
        processor = VoiceProcessor(
            model_dir="pretrained_models/Spark-TTS-0.5B",  # 添加模型路径
            whisper_model_size=args.whisper_model_size,
            compute_type=args.compute_type,
            language=args.language,
            output_dir=args.output_dir,
        )

        # 使用命令行参数进行语音处理
        recognition_results, output_path = processor.process_voice(
            source_audio=args.input1,
            target_text=args.input2,
            voice_speed=args.voice_speed,
        )

        print(f"\n合成结果保存在: {output_path}")
    except Exception as e:
        print(f"处理失败: {str(e)}")
