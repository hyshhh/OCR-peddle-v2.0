<h1 align="center">Ship Hull Agent</h1>

<p align="center">
  <strong>基于 LangChain + FAISS + YOLO + Qwen3 VLM + PaddleOCR 的智能船弦号识别系统</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-0.3-orange" alt="LangChain">
  <img src="https://img.shields.io/badge/YOLOv8-green" alt="YOLO">
  <img src="https://img.shields.io/badge/Qwen3-VL-purple" alt="Qwen3">
  <img src="https://img.shields.io/badge/PaddleOCR-3.5-blue" alt="PaddleOCR">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p align="center">
  两大核心模块：<strong>Agent 对话检索</strong>（精确匹配 + RAG 语义搜索）+ <strong>Pipeline 视频处理</strong>（实时检测 + 跟踪 + 识别 + 弦号定位）
</p>

---

## Features

<table>
<tr>
<td width="50%">

### Agent 对话检索

| 功能 | 说明 |
|------|------|
| 精确弦号匹配 | 输入弦号直接查字典，O(1) 响应 |
| RAG 语义检索 | FAISS 向量库对船描述做语义相似度匹配 |
| 智能路由 | 有弦号先精确查，查不到再语义检索 |
| 阈值过滤 | 语义检索结果低于置信度阈值自动过滤 |
| 向量库持久化 | FAISS 索引首次构建后缓存磁盘，后续直接加载 |
| 自动变更检测 | MD5 哈希比对 CSV，数据变更自动重建向量库 |
| 批量建库 | 视觉模型自动识别图片生成弦号+描述 |

</td>
<td width="50%">

### Pipeline 视频处理

| 功能 | 说明 |
|------|------|
| YOLO 船只检测 | ultralytics YOLO + ByteTrack / BoTSORT 追踪 |
| 跟踪 ID 绑定 | track ID 绑定唯一弦号，跟踪持续则沿用 |
| 定时刷新 | 每隔 N 帧自动重新识别已跟踪船只 |
| 弦号定位 | PaddleOCR TextDetection 在 crop 中定位文字区域 |
| Qwen3 VLM 识别 | 视觉大模型进行弦号识别与描述生成 |
| 级联/并发双模式 | 级联同步等待；并发双层架构（帧级队列 + crop 级并发） |
| 智能跳帧 | YOLO 检测频率与推理频率独立控制 |
| Demo 可视化 | 实时显示检测框、跟踪 ID、识别结果、弦号定位框 |

</td>
</tr>
</table>

---

## Architecture

```
                           ┌─────────────────────────────────────────────────────────────────┐
                           │                    Pipeline 视频处理流水线                       │
                           └─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │   视频输入    │────▶│  YOLO 检测+跟踪   │────▶│  裁剪船只 (crop)  │────▶│  弦号定位（可选） │
  │ 文件/相机/流  │     │  ByteTrack       │     │  尺寸归一化       │     │  PaddleOCR       │
  └──────────────┘     └──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                                                  │
                    ┌──────────────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │                              三步链路推理                                                │
  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │
  │  │  VLM 识别     │───▶│  精确查库     │───▶│  语义检索     │───▶│  绑定结果     │       │
  │  │  弦号+描述    │    │  O(1) 匹配    │    │  FAISS 向量库 │    │  track 绑定   │       │
  │  └───────────────┘    └───────────────┘    └───────────────┘    └───────────────┘       │
  └─────────────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │  渲染检测框 + 弦号定位框 + 识别结果 + FPS HUD ──▶ 输出视频 / Demo 窗口                    │
  └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/hyshhh/OCR-peddle-v2.0.git
cd OCR-peddle

# 安装依赖
pip install -e .

# 开发模式（含测试依赖）
pip install -e ".[dev]"
```

### 2. Start VLM Service

使用 vLLM 部署 Qwen3 VLM（兼容 OpenAI API 格式）：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3-VL-4B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 10240 \
  --port 7890 \
  --gpu-memory-utilization 0.15 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

### 3. Start Embedding Service (Optional, for RAG)

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3-Embedding-0.6B \
  --api-key abc123 \
  --served-model-name Qwen3-Embedding-0.6B \
  --convert embed \
  --gpu-memory-utilization 0.08 \
  --max-model-len 2048 \
  --port 7891
```

### 4. Configure

```bash
cp .env.example .env
```

Edit `config.yaml`:

```yaml
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "abc123"
  base_url: "http://localhost:7890/v1"
  temperature: 0.0

embed:
  model: "Qwen3-Embedding-0.6B"
  api_key: "abc123"
  base_url: "http://localhost:7891/v1"

retrieval:
  top_k: 3
  score_threshold: 0.5
```

### 5. Run

```bash
# Agent 单次查询
ship-hull "帮我查一下弦号0014是什么船"

# Agent 交互模式
ship-hull --interactive

# Pipeline 处理视频
python -m pipeline.cli video.mp4 --demo --output result.mp4
```

---

## Usage Examples

### Agent Query

**精确匹配**

```bash
$ ship-hull "帮我查一下弦号0014是什么船"

识别结果：弦号 0014，描述：白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪
```

**语义检索（弦号不存在）**

```bash
$ ship-hull "弦号9999，这是一艘大型白色邮轮，船身有蓝色条纹装饰，有三个烟囱"

未找到对应弦号，根据描述检索到最相似的船：
1. 弦号 0123，描述：白色邮轮，船身有红蓝条纹装饰，三座烟囱（相似度：0.9234）
2. 弦号 0014，描述：白色大型客轮，上层建筑为蓝色涂装（相似度：0.7521）
```

**详细模式（调试调用链）**

```bash
$ ship-hull --verbose "我看到一艘灰色的军舰，外形很隐身"

+------------------- Agent 调用链 -------------------+
| # | 类型     | 内容                                   |
+---+----------+----------------------------------------+
| 0 | human    | 我看到一艘灰色的军舰，外形很隐身         |
| 1 | ai       | -> lookup_by_hull_number({"hull_number" |
| 2 | tool     | <- {"found": false, "hull_number": ""}  |
| 3 | ai       | -> retrieve_by_description({"target_de" |
| 4 | tool     | <- {"results": [{"hull_number": "0256"  |
| 5 | ai       | 未找到对应弦号，根据描述检索到最相似的船  |
+----------------------------------------------------+
```

### Pipeline Commands

```bash
# 处理视频文件
python -m pipeline.cli video.mp4

# USB 相机
python -m pipeline.cli 0

# RTSP 视频流
python -m pipeline.cli rtsp://192.168.1.100/stream

# Demo 可视化 + 输出结果视频
python -m pipeline.cli video.mp4 --demo --output result.mp4

# 并发模式（高吞吐）
python -m pipeline.cli video.mp4 --concurrent --max-concurrent 8

# 启用弦号定位（PaddleOCR）
python -m pipeline.cli video.mp4 --hull-locator --demo

# 简略提示词 + 每 5 帧处理一次（提速）
python -m pipeline.cli video.mp4 --prompt-mode brief --process-every 5
```

---

## Configuration

### Config Reference

```yaml
# ═══════════════════════════════════════════
# LLM / Embedding / RAG
# ═══════════════════════════════════════════

llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "abc123"
  base_url: "http://localhost:7890/v1"
  temperature: 0.0

embed:
  model: "Qwen3-Embedding-0.6B"
  api_key: "abc123"
  base_url: "http://localhost:7891/v1"

retrieval:
  top_k: 3
  score_threshold: 0.5

vector_store:
  persist_path: "./vector_store"
  auto_rebuild: false

# ═══════════════════════════════════════════
# Pipeline 视频处理流水线
# ═══════════════════════════════════════════

pipeline:
  # 模式
  concurrent_mode: false          # false=级联 true=并发
  max_concurrent: 4               # 并发模式最大推理数
  max_queued_frames: 30           # 最大队列深度（防 OOM）

  # 帧控制
  process_every_n_frames: 15      # 每 N 帧触发推理
  detect_every_n_frames: 5        # 每 N 帧做 YOLO 检测

  # YOLO
  yolo_model: "yolov8n.pt"
  device: ""                      # "" 自动 / "cpu" / "0" GPU
  conf_threshold: 0.25
  detect_classes: [8]             # COCO: 8=boat

  # 追踪
  tracker: "bytetrack"
  tracker_params:
    track_high_thresh: 0.5
    track_low_thresh: 0.05
    new_track_thresh: 0.6
    track_buffer: 90
    match_thresh: 0.5
  max_stale_frames: 30

  # Crop
  crop_min_size: 512
  crop_max_size: 1024

  # 其他
  prompt_mode: "detailed"         # detailed / brief
  enable_refresh: false
  gap_num: 150
  save_screenshots: true
  output_dir: "./output"
  demo: false

# ═══════════════════════════════════════════
# 弦号定位（PaddleOCR TextDetection）
# ═══════════════════════════════════════════

hull_locator:
  # 开关
  enabled: true

  # 后处理过滤
  score_threshold: 0.0            # 模型返回后过滤（0.0~1.0）
  min_area: 0                     # 最小区域面积（像素²）

  # PaddleOCR 内部参数
  text_det_thresh: 0.3            # 像素级阈值：概率图中超过此值的像素视为文字
  text_det_box_thresh: 0.5        # 框级阈值：候选框平均分数低于此值被过滤
  text_det_unclip_ratio: 1.5      # 膨胀系数：值越大检测框越大
  text_det_max_side_limit: 960    # 输入图像最大边长（像素）
  use_dilation: true              # 是否使用膨胀操作
  det_db_score_mode: "fast"       # 评分模式：fast / slow
```

---

## Project Structure

```
OCR-peddle/
├── config.py                # 配置读取：config.yaml + 内置默认值
├── config.yaml              # 全局配置文件
├── build_db.py              # 批量建库脚本（图片 → 弦号+描述 → CSV）
│
├── database/
│   └── __init__.py          # ShipDatabase：CSV + FAISS 向量库 + 自动变更检测
│
├── tools/
│   └── __init__.py          # LangChain @tool：recognize / lookup / retrieve
│
├── agent/
│   ├── __init__.py          # ShipHullAgent：ReAct Agent
│   └── result.py            # AgentResult 数据结构
│
├── cli/
│   └── __init__.py          # Rich CLI：单次查询 / 交互 REPL
│
├── pipeline/
│   ├── __main__.py          # python -m pipeline 入口
│   ├── cli.py               # 命令行参数解析
│   ├── pipeline.py          # 主流水线编排（ShipPipeline）
│   ├── detector.py          # YOLO 检测 + ByteTrack 跟踪
│   ├── locator.py           # PaddleOCR TextDetection 弦号定位
│   ├── agent_inference.py   # Qwen3 VLM 推理
│   ├── tracker.py           # 跟踪状态管理（线程安全）
│   ├── fps.py               # FPS 统计 + 延迟监控
│   ├── video_input.py       # 视频/相机/视频流统一输入
│   ├── demo.py              # Demo 可视化渲染
│   └── output.py            # 截图保存
│
├── data/ships.csv           # 船只数据库
├── tests/                   # 单元测试
├── pyproject.toml           # 项目元数据 + 依赖
└── requirements.txt         # 依赖清单
```

---

## Tech Stack

<table>
<tr>
<td><strong>LLM 编排</strong></td>
<td>LangChain + LangGraph</td>
<td>ReAct Agent 模式</td>
</tr>
<tr>
<td><strong>向量库</strong></td>
<td>FAISS (faiss-cpu)</td>
<td>语义检索索引</td>
</tr>
<tr>
<td><strong>视觉模型</strong></td>
<td>Qwen3 VLM</td>
<td>船只图像弦号识别</td>
</tr>
<tr>
<td><strong>弦号定位</strong></td>
<td>PaddleOCR TextDetection</td>
<td>crop 内文字区域检测</td>
</tr>
<tr>
<td><strong>目标检测</strong></td>
<td>ultralytics YOLO</td>
<td>船只检测</td>
</tr>
<tr>
<td><strong>跟踪</strong></td>
<td>ByteTrack (YOLO 内置)</td>
<td>多目标跟踪</td>
</tr>
<tr>
<td><strong>视频处理</strong></td>
<td>OpenCV (cv2)</td>
<td>视频读写、图像处理</td>
</tr>
<tr>
<td><strong>并发</strong></td>
<td>threading + queue.Queue</td>
<td>级联/并发双模式</td>
</tr>
</table>

---

## License

MIT
