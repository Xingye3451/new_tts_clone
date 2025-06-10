# 声音合成 gRPC 服务

## 服务描述

这是一个基于 gRPC 的声音合成服务，提供音色克隆（声音复制）和语音合成功能，支持实时进度通知。该服务运行在 Docker 容器中，通过内部网络 `ai-platform-network` 进行通信，不对外暴露端口，确保安全性。

## 主要功能

- **音色克隆**：使用源音频的声音特征合成新的文本内容
- **指定音色合成**：使用预设声音合成文本
- **实时进度通知**：通过流式 gRPC 提供进度更新
- **音频识别**：自动识别源音频内容（音色克隆时）
- **多任务并行处理**：支持多个任务同时处理
- **Docker 容器化运行**：稳定可靠，资源隔离
- **GPU 加速**：支持 CUDA 加速，提高处理速度

## 环境要求

- Docker 和 Docker Compose
- NVIDIA GPU 和 NVIDIA Container Toolkit（推荐）
- 预先准备的模型文件（可选，首次运行时会自动下载）
  - Spark-TTS 模型
  - Whisper 模型

## 模型文件

服务使用两种模型：

1. **Spark-TTS 模型**：用于声音合成，默认位置：`/data/models/Spark-TTS-0.5B`
2. **Whisper 模型**：用于音频识别，默认位置：`/root/.cache/whisper`

如果模型文件不存在，服务会在首次运行时自动下载。

## 安装部署

### 1. 创建 Docker 网络

```bash
docker network create ai-platform-network
```

### 2. 准备目录结构

```bash
mkdir -p outputs pretrained_models whisper_model
```

### 3. 构建并启动服务

```bash
docker-compose up -d
```

服务将在内部网络 `ai-platform-network` 中的 `50051` 端口运行，可以通过服务名 `voice-service` 访问。

## 服务接口

### 1. 音色克隆

```protobuf
rpc CloneVoiceAsync(CloneVoiceRequest) returns (CloneVoiceResponse)
```

#### 请求参数

- `source_audio`: 源音频文件路径
- `target_text`: 目标文本（可选，如果不提供则使用识别结果）
- `voice_speed`: 语音速度（1.0 为正常速度）
- `output_dir`: 输出目录（可选）

#### 响应

- `task_id`: 任务 ID
- `output_path`: 输出文件路径
- `status`: 任务状态

### 2. 指定音色合成

```protobuf
rpc SynthesizeAsync(SynthesizeRequest) returns (SynthesizeResponse)
```

#### 请求参数

- `text`: 要合成的文本
- `voice_name`: 声音名称（可选）
- `voice_speed`: 语音速度（1.0 为正常速度）
- `output_dir`: 输出目录（可选）

#### 响应

- `task_id`: 任务 ID
- `output_path`: 输出文件路径
- `status`: 任务状态

### 3. 获取进度更新

```protobuf
rpc GetProgressUpdates(ProgressRequest) returns (stream ProgressResponse)
```

#### 请求参数

- `task_id`: 任务 ID

#### 响应流

- `task_id`: 任务 ID
- `progress`: 进度百分比
- `status`: 任务状态
- `message`: 进度消息
- `is_finished`: 是否完成
- `output_path`: 输出文件路径（仅在完成时返回）
- `segments`: 识别结果片段（仅在音色克隆时返回）

## 客户端示例

服务提供了一个命令行客户端示例，可以用来测试服务功能。

### 音色克隆示例

```bash
python grpc_client_example.py clone \
  --source /path/to/audio.wav \
  --text "要合成的文本内容" \
  --speed 1.0 \
  --server voice-service:50051
```

如果不提供 `--text` 参数，将使用源音频的识别结果作为合成文本。

### 指定音色合成示例

```bash
python grpc_client_example.py tts \
  --text "要合成的文本内容" \
  --voice "voice_name" \
  --speed 1.0 \
  --server voice-service:50051
```

## 集成到其他应用

### Python 集成示例

```python
import grpc
from grpc_service import voice_service_pb2
from grpc_service import voice_service_pb2_grpc

def clone_voice(source_audio, target_text, voice_speed=1.0):
    # 创建通道
    channel = grpc.insecure_channel('voice-service:50051')
    stub = voice_service_pb2_grpc.VoiceServiceStub(channel)

    try:
        # 创建请求
        request = voice_service_pb2.CloneVoiceRequest(
            source_audio=source_audio,
            target_text=target_text,
            voice_speed=voice_speed
        )

        # 发送请求
        response = stub.CloneVoiceAsync(request)
        task_id = response.task_id

        # 获取进度更新
        progress_request = voice_service_pb2.ProgressRequest(task_id=task_id)
        for progress in stub.GetProgressUpdates(progress_request):
            print(f"进度: {progress.progress}% - {progress.message}")
            if progress.is_finished:
                return progress.output_path
    finally:
        channel.close()
```

### 其他语言

对于其他编程语言，需要：

1. 使用 protoc 根据 `voice_service.proto` 文件生成对应语言的代码
2. 创建 gRPC 客户端和请求对象
3. 调用相应的方法并处理响应

## 配置参数

服务支持以下环境变量进行配置：

| 环境变量     | 说明             | 默认值                      |
| ------------ | ---------------- | --------------------------- |
| OUTPUT_DIR   | 输出目录         | /data/outputs               |
| MODEL_DIR    | 模型目录         | /data/models/Spark-TTS-0.5B |
| WHISPER_SIZE | Whisper 模型大小 | small                       |
| COMPUTE_TYPE | 计算类型         | int8                        |
| LANGUAGE     | 语言             | zh                          |

## 手工测试步骤

### 1. 创建测试网络

```bash
docker network create ai-platform-network
```

### 2. 启动服务

```bash
docker-compose up -d
docker-compose ps  # 检查服务状态
```

### 3. 创建测试客户端容器

```bash
docker run -it --network ai-platform-network --rm python:3.9 bash
```

### 4. 在容器中测试

```bash
# 安装依赖
pip install grpcio grpcio-tools

# 创建一个简单的测试脚本
cat > test_client.py << 'EOF'
import grpc
import time
import sys

# 生成gRPC代码（需要先获取proto文件）
with open('voice_service.proto', 'w') as f:
    f.write("""
syntax = "proto3";

package voice_service;

service VoiceService {
  rpc CloneVoiceAsync(CloneVoiceRequest) returns (CloneVoiceResponse) {}
  rpc SynthesizeAsync(SynthesizeRequest) returns (SynthesizeResponse) {}
  rpc GetProgressUpdates(ProgressRequest) returns (stream ProgressResponse) {}
}

message CloneVoiceRequest {
  string source_audio = 1;
  string target_text = 2;
  float voice_speed = 3;
  string output_dir = 4;
}

message CloneVoiceResponse {
  string task_id = 1;
  string output_path = 2;
  string status = 3;
}

message SynthesizeRequest {
  string text = 1;
  string voice_name = 2;
  float voice_speed = 3;
  string output_dir = 4;
}

message SynthesizeResponse {
  string task_id = 1;
  string output_path = 2;
  string status = 3;
}

message ProgressRequest {
  string task_id = 1;
}

message ProgressResponse {
  string task_id = 1;
  int32 progress = 2;
  string status = 3;
  string message = 4;
  bool is_finished = 5;
  string output_path = 6;
  repeated TranscriptionSegment segments = 7;
}

message TranscriptionSegment {
  string text = 1;
  float start = 2;
  float end = 3;
}
    """)

# 生成gRPC代码
import subprocess
subprocess.run(['python', '-m', 'grpc_tools.protoc', '-I.', '--python_out=.', '--grpc_python_out=.', 'voice_service.proto'])

# 导入生成的模块
import voice_service_pb2
import voice_service_pb2_grpc

# 创建gRPC通道
channel = grpc.insecure_channel('voice-service:50051')
stub = voice_service_pb2_grpc.VoiceServiceStub(channel)

# 测试TTS功能
try:
    request = voice_service_pb2.SynthesizeRequest(
        text="这是一个测试音频，用于验证gRPC服务是否正常工作。",
        voice_speed=1.0
    )

    response = stub.SynthesizeAsync(request)
    print(f"任务ID: {response.task_id}")
    print(f"输出路径: {response.output_path}")

    # 获取进度更新
    progress_request = voice_service_pb2.ProgressRequest(task_id=response.task_id)
    for progress in stub.GetProgressUpdates(progress_request):
        print(f"进度: {progress.progress}% - {progress.message}")
        if progress.is_finished:
            print(f"完成: {progress.output_path}")
            break
        time.sleep(0.5)

except grpc.RpcError as e:
    print(f"RPC错误: {e.details()}")
    sys.exit(1)
EOF

# 运行测试脚本
python test_client.py
```

### 5. 从主机测试（可选）

```bash
# 获取服务容器IP
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' spark-tts-grpc

# 修改客户端脚本中的服务器地址，然后运行
python grpc_client_example.py tts \
  --text "测试文本" \
  --server <容器IP>:50051
```

## 注意事项

1. 确保有足够的存储空间用于存放模型文件和输出音频
2. GPU 加速需要安装 NVIDIA Container Toolkit
3. 首次运行时，如果模型不存在，服务会自动下载，可能需要较长时间
4. 处理长文本时可能需要更多内存和处理时间
5. 音频文件路径必须是容器内可访问的路径
6. 服务运行在内部网络，不对外暴露端口，确保安全性
