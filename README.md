<div align="center">
    <h1>
    Spark-TTS
    </h1>
    <p>
    Official PyTorch code for inference of <br>
    <b><em>Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens</em></b>
    </p>
    <p>
    <img src="src/logo/SparkTTS.jpg" alt="Spark-TTS Logo" style="width: 200px; height: 200px;">
    </p>
        <p>
        <img src="src/logo/HKUST.jpg" alt="Institution 1" style="width: 200px; height: 60px;">
        <img src="src/logo/mobvoi.jpg" alt="Institution 2" style="width: 200px; height: 60px;">
        <img src="src/logo/SJU.jpg" alt="Institution 3" style="width: 200px; height: 60px;">
    </p>
    <p>
        <img src="src/logo/NTU.jpg" alt="Institution 4" style="width: 200px; height: 60px;">
        <img src="src/logo/NPU.jpg" alt="Institution 5" style="width: 200px; height: 60px;">
        <img src="src/logo/SparkAudio2.jpg" alt="Institution 6" style="width: 200px; height: 60px;">
    </p>
    <p>
    </p>
    <a href="https://arxiv.org/pdf/2503.01710"><img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="paper"></a>
    <a href="https://sparkaudio.github.io/spark-tts/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://huggingface.co/SparkAudio/Spark-TTS-0.5B"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="Hugging Face"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Python-3.12+-orange" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/PyTorch-2.5+-brightgreen" alt="python"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>

## Spark-TTS 🔥

### Overview

Spark-TTS is an advanced text-to-speech system that uses the power of large language models (LLM) for highly accurate and natural-sounding voice synthesis. It is designed to be efficient, flexible, and powerful for both research and production use.

### Key Features

- **Simplicity and Efficiency**: Built entirely on Qwen2.5, Spark-TTS eliminates the need for additional generation models like flow matching. Instead of relying on separate models to generate acoustic features, it directly reconstructs audio from the code predicted by the LLM. This approach streamlines the process, improving efficiency and reducing complexity.
- **High-Quality Voice Cloning**: Supports zero-shot voice cloning, which means it can replicate a speaker's voice even without specific training data for that voice. This is ideal for cross-lingual and code-switching scenarios, allowing for seamless transitions between languages and voices without requiring separate training for each one.
- **Bilingual Support**: Supports both Chinese and English, and is capable of zero-shot voice cloning for cross-lingual and code-switching scenarios, enabling the model to synthesize speech in multiple languages with high naturalness and accuracy.
- **Controllable Speech Generation**: Supports creating virtual speakers by adjusting parameters such as gender, pitch, and speaking rate.
- **GPU Acceleration**: Fully optimized for GPU acceleration, supporting both CUDA and ROCm backends for maximum performance.
- **Production Ready**: Includes Docker support and API services for easy deployment in production environments.

---

### System Requirements

- **Hardware Requirements**:
  - CPU: 4+ cores recommended
  - RAM: 16GB minimum, 32GB recommended
  - GPU: NVIDIA GPU with 8GB+ VRAM (for GPU acceleration)
  - Storage: 20GB+ free space

- **Software Requirements**:
  - Operating System: Ubuntu 20.04+ / Windows 10+ / macOS 12+
  - CUDA Toolkit 11.8+ (for GPU support)
  - Python 3.12+
  - PyTorch 2.5+

### Quick Start Guide

1. **Environment Setup**:
```bash
# Create and activate conda environment
conda create -n sparktts python=3.12 -y
conda activate sparktts

# Install dependencies
pip install -r requirements.txt

# For Chinese users (optional)
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

2. **Model Download**:
```bash
# Option 1: Using Python
python -c "from huggingface_hub import snapshot_download; snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='pretrained_models/Spark-TTS-0.5B')"

# Option 2: Using Git LFS
git lfs install
git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B
```

3. **Basic Usage**:
```bash
# Simple text-to-speech
python -m cli.inference \
    --text "Hello, this is a test." \
    --device 0 \
    --save_dir "outputs" \
    --model_dir pretrained_models/Spark-TTS-0.5B

# Voice cloning with reference audio
python -m cli.inference \
    --text "Clone this voice." \
    --device 0 \
    --save_dir "outputs" \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --prompt_text "Reference audio transcript" \
    --prompt_speech_path "path/to/reference.wav"
```

### Advanced Features

1. **Voice Control Parameters**:
- Gender: male/female
- Age: young/middle/old
- Pitch: low/normal/high
- Speed: slow/normal/fast
- Style: normal/happy/sad/angry

Example:
```bash
python -m cli.inference \
    --text "Custom voice test." \
    --gender female \
    --pitch high \
    --speed normal \
    --style happy
```

2. **Batch Processing**:
```bash
python -m cli.batch_inference \
    --input_file texts.txt \
    --output_dir outputs \
    --model_dir pretrained_models/Spark-TTS-0.5B
```

3. **API Service Configuration**:
```bash
# Start API service
./api/run_api.sh

# Custom port and host
export SPARK_TTS_PORT=8000
export SPARK_TTS_HOST=0.0.0.0
./api/run_api.sh
```

### Docker Deployment

1. **Build Images**:
```bash
# Build all images
./docker_builder.sh

# Build specific version
./docker_builder.sh full  # or lite
```

2. **Run Services**:
```bash
# Start all services
docker compose up -d

# Start specific service
docker compose up -d api
docker compose up -d webui
```

3. **Environment Variables**:
```bash
# API configuration
SPARK_TTS_PORT=7860
SPARK_TTS_HOST=0.0.0.0
SPARK_TTS_MODEL_DIR=/app/pretrained_models/Spark-TTS-0.5B
SPARK_TTS_DEVICE=cuda

# Resource limits
SPARK_TTS_MAX_MEMORY=8g
SPARK_TTS_CPU_COUNT=4
```

### Troubleshooting

1. **Common Issues**:
- **CUDA Out of Memory**: Reduce batch size or use smaller model
- **Missing Dependencies**: Check requirements.txt and CUDA version
- **Audio Quality Issues**: Check input audio format and sample rate

2. **Performance Optimization**:
- Use GPU acceleration when possible
- Enable batch processing for multiple requests
- Optimize model loading and caching

3. **Error Reporting**:
Please include the following when reporting issues:
- System information (OS, Python version, GPU)
- Complete error message
- Minimal reproducible example
- Input text and parameters used

### Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Create a pull request

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Contact

- Issues: Please use GitHub Issues
- Email: [project-email@example.com]
- WeChat Group: [Contact maintainers for invitation]

---

<table align="center">
  <tr>
    <td align="center"><b>Inference Overview of Voice Cloning</b><br><img src="src/figures/infer_voice_cloning.png" width="80%" /></td>
  </tr>
  <tr>
    <td align="center"><b>Inference Overview of Controlled Generation</b><br><img src="src/figures/infer_control.png" width="80%" /></td>
  </tr>
</table>

## 🚀 News

- **[2025-03-04]** Our paper on this project has been published! You can read it here: [Spark-TTS](https://arxiv.org/pdf/2503.01710).

## Install

**Clone and Install**

Here are instructions for installing on Linux. If you're on Windows, please refer to the [Windows Installation Guide](https://github.com/SparkAudio/Spark-TTS/issues/5).  
_(Thanks to [@AcTePuKc](https://github.com/AcTePuKc) for the detailed Windows instructions!)_

- Clone the repo

```sh
git clone https://github.com/SparkAudio/Spark-TTS.git
cd Spark-TTS
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

```sh
conda create -n sparktts -y python=3.12
conda activate sparktts
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**Model Download**

Download via python:

```python
from huggingface_hub import snapshot_download

snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
```

Download via git clone:

```sh
mkdir -p pretrained_models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B
```

**Basic Usage**

You can simply run the demo with the following commands:

```sh
cd example
bash infer.sh
```

Alternatively, you can directly execute the following command in the command line to perform inference：

```sh
python -m cli.inference \
    --text "text to synthesis." \
    --device 0 \
    --save_dir "path/to/save/audio" \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --prompt_text "transcript of the prompt audio" \
    --prompt_speech_path "path/to/prompt_audio"
```

**Web UI Usage**

You can start the UI interface by running `python webui.py --device 0`, which allows you to perform Voice Cloning and Voice Creation. Voice Cloning supports uploading reference audio or directly recording the audio.

|           **Voice Cloning**            |             **Voice Creation**             |
| :------------------------------------: | :----------------------------------------: |
| ![Image 1](src/figures/gradio_TTS.png) | ![Image 2](src/figures/gradio_control.png) |

**Optional Methods**

For additional CLI and Web UI methods, including alternative implementations and extended functionalities, you can refer to:

- [CLI and UI by AcTePuKc](https://github.com/SparkAudio/Spark-TTS/issues/10)

**API Service**

Spark-TTS provides a FastAPI-based web API service for seamless integration with other applications.

1. **Running the API service in conda environment**:

   ```sh
   # Make sure you're in the Spark-TTS conda environment
   conda activate sparktts

   # Execute from the project root directory
   ./api/run_api.sh
   ```

   The API will be available at http://localhost:7860 by default.

2. **Docker support**:
   You can build and run the Spark-TTS API using the provided build script:

   ```sh
   # Build Docker images (both full and lite versions)
   chmod +x docker_builder.sh
   ./docker_builder.sh

   # Run the API service in the background
   docker compose up -d api
   # OR for the lite version with mounted models
   docker compose up -d api-lite

   # Run the WebUI service in the background
   docker compose up -d webui
   # OR for the lite version with mounted models
   docker compose up -d webui-lite

   # To check running containers
   docker compose ps

   # To stop services
   docker compose down
   ```

   > **Note**: If you encounter YAML errors like `mapping key "<<" already defined`, it might be due to compatibility issues with YAML merge keys in your Docker Compose version. You can either:
   >
   > 1. Update Docker to the latest version
   > 2. Modify the docker-compose.yml file to use a different syntax for environment variable inheritance
   > 3. Use the Docker CLI directly: `docker run -p 7860:7860 --gpus all spark-tts:latest-full`

   For more customization options, see the environment variables in the docker-compose.yml file.

3. **Client Example**:
   The repository includes an example client script that demonstrates how to interact with the API:

   ```sh
   # Note: The example client requires librosa, which is not in requirements.txt
   pip install librosa

   # Basic usage
   python api/example_client.py --text "Text to synthesize"

   # Voice cloning with reference audio
   python api/example_client.py --text "This is voice cloning" --prompt_audio example/prompt_audio.wav

   # Voice creation with parameters
   python api/example_client.py --text "This is voice creation" --gender female --pitch high --speed moderate
   ```

For more detailed information about the API service, including all available endpoints and parameters, please refer to the [API README](api/README.md).

## **Demos**

Here are some demos generated by Spark-TTS using zero-shot voice cloning. For more demos, visit our [demo page](https://sparkaudio.github.io/spark-tts/).

---

<table>
<tr>
<td align="center">
    
**Donald Trump**
</td>
<td align="center">
    
**Zhongli (Genshin Impact)**
</td>
</tr>

<tr>
<td align="center">

[Donald Trump](https://github.com/user-attachments/assets/fb225780-d9fe-44b2-9b2e-54390cb3d8fd)

</td>
<td align="center">
    
[Zhongli](https://github.com/user-attachments/assets/80eeb9c7-0443-4758-a1ce-55ac59e64bd6)

</td>
</tr>
</table>

---

<table>

<tr>
<td align="center">
    
**陈鲁豫 Chen Luyu**
</td>
<td align="center">
    
**杨澜 Yang Lan**
</td>
</tr>

<tr>
<td align="center">
    
[陈鲁豫Chen_Luyu.webm](https://github.com/user-attachments/assets/5c6585ae-830d-47b1-992d-ee3691f48cf4)
</td>
<td align="center">
    
[Yang_Lan.webm](https://github.com/user-attachments/assets/2fb3d00c-abc3-410e-932f-46ba204fb1d7)
</td>
</tr>
</table>

---

<table>
<tr>
<td align="center">
    
**余承东 Richard Yu**
</td>
<td align="center">
    
**马云 Jack Ma**
</td>
</tr>

<tr>
<td align="center">

[Yu_Chengdong.webm](https://github.com/user-attachments/assets/78feca02-84bb-4d3a-a770-0cfd02f1a8da)

</td>
<td align="center">
    
[Ma_Yun.webm](https://github.com/user-attachments/assets/2d54e2eb-cec4-4c2f-8c84-8fe587da321b)

</td>
</tr>
</table>

---

<table>
<tr>
<td align="center">
    
**刘德华 Andy Lau**
</td>
<td align="center">

**徐志胜 Xu Zhisheng**

</td>
</tr>

<tr>
<td align="center">

[Liu_Dehua.webm](https://github.com/user-attachments/assets/195b5e97-1fee-4955-b954-6d10fa04f1d7)

</td>
<td align="center">
    
[Xu_Zhisheng.webm](https://github.com/user-attachments/assets/dd812af9-76bd-4e26-9988-9cdb9ccbb87b)

</td>
</tr>
</table>

---

<table>
<tr>
<td align="center">
    
**哪吒 Nezha**
</td>
<td align="center">
    
**李靖 Li Jing**
</td>
</tr>

<tr>
<td align="center">

[Ne_Zha.webm](https://github.com/user-attachments/assets/8c608037-a17a-46d4-8588-4db34b49ed1d)

</td>
<td align="center">

[Li_Jing.webm](https://github.com/user-attachments/assets/aa8ba091-097c-4156-b4e3-6445da5ea101)

</td>
</tr>
</table>

## To-Do List

- [x] Release the Spark-TTS paper.
- [ ] Release the training code.
- [ ] Release the training dataset, VoxBox.

## Citation

```
@misc{wang2025sparktts,
      title={Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens},
      author={Xinsheng Wang and Mingqi Jiang and Ziyang Ma and Ziyu Zhang and Songxiang Liu and Linqin Li and Zheng Liang and Qixi Zheng and Rui Wang and Xiaoqin Feng and Weizhen Bian and Zhen Ye and Sitong Cheng and Ruibin Yuan and Zhixian Zhao and Xinfa Zhu and Jiahao Pan and Liumeng Xue and Pengcheng Zhu and Yunlin Chen and Zhifei Li and Xie Chen and Lei Xie and Yike Guo and Wei Xue},
      year={2025},
      eprint={2503.01710},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.01710},
}
```

## ⚠️ Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

- Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

- The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.
