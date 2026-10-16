# ComfyUI MiniMax-H3 MLX

把 [Argus-AiTeam/minimax-h3-mac](https://github.com/Argus-AiTeam/minimax-h3-mac) 封装为 ComfyUI 自定义节点，在 Apple Silicon Mac 上通过 MLX 生成带立体声音频的视频。

> 这不是 CUDA/PyTorch 移植。推理仍由上游 MLX 管线执行，因此只支持 Apple Silicon（M 系列）Mac。

## MiniMax Music 3 (MLX)

This fork adds two nodes that generate full songs with
[MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) through the
native MLX port in [`mlx-audio`](https://github.com/Blaizzy/mlx-audio)
(`mlx_audio.music`). ComfyUI's built-in Music3 support runs on PyTorch/MPS; these
nodes run the whole pipeline (Qwen3 AR song writer, RVQ depth decoder,
flow-matching DiT, vocoder) on MLX instead.

- **Load MiniMax Music3 (MLX)**: pick an MLX conversion from
  `ComfyUI/models/minimax_music3/` or an `mlx-community` preset. Missing presets
  download into that folder when `allow_download` is on. `custom_path` accepts
  any MLX Music3 directory.
- **Generate MiniMax Music3 Song**: caption + lyrics in, ComfyUI `AUDIO`
  (stereo, 44.1 kHz) out. Wire it to the core Preview Audio / Save Audio nodes.

```
Load MiniMax Music3 (MLX) -> Generate MiniMax Music3 Song -> Save Audio
```

`example_workflows/minimax_music3_mlx_text_to_song.json` contains that graph and
shows up under ComfyUI's workflow templates for this node pack.

### Weights

| Preset | Notes |
| --- | --- |
| `mlx-community/MiniMax-Music3-mxfp8` | Default; best balance of memory and lyric fidelity |
| `mlx-community/MiniMax-Music3-8bit` / `-6bit` / `-4bit` | Affine quants |
| `mlx-community/MiniMax-Music3-mxfp4` | Experimental; can drop or alter lyrics |
| `mlx-community/MiniMax-Music3-bf16` | Dense reference |

Accept the [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE)
first. To download ahead of time instead of from the node:

```
hf download mlx-community/MiniMax-Music3-mxfp8 \
  --local-dir ComfyUI/models/minimax_music3/MiniMax-Music3-mxfp8
```

A model directory needs `config.json`, the `*.safetensors` weights and the
official `tokenizer/`. The loader refuses directories without the tokenizer,
because `mlx-audio` would otherwise fall back to its test tokenizer and
produce noise.

### Generation

- `caption`: style, mood, BPM, instrumentation, vocal, arrangement.
- `lyrics`: section tags (`[intro]`, `[verse]`, `[chorus]`, `[bridge]`, `[outro]`)
  on their own lines. Lyrics are required; use `[instrumental]` for no vocals.
- `duration_seconds`: upper bound, 1–360 s. The model may end the song earlier.
- `steps`: flow-matching steps per 8 s chunk, 1–30.
- `seed`: drives the autoregressive stage, which decides most of the song.
  Change it to reroll the composition.

### Who loads the model, and when

The loader only resolves (or downloads) the weights. The model is read into
unified memory on the first generation and then stays resident inside the
loader's cached output, so later runs skip loading. It is freed when
`unload_after` is on, when the loader inputs change, or when ComfyUI drops its
node cache. ComfyUI's own model manager does not track MLX memory, so
"Unload models" in the UI does not release it. Unified memory is shared with
PyTorch and any H3 pipeline, so avoid keeping both models resident on smaller Macs.

Progress is reported per autoregressive frame (most of the runtime) and per
diffusion chunk, and Cancel takes effect between them.

### Tests

```
pip install "mlx[cpu]" "mlx-audio>=0.5.9" pytest   # mlx (no extra) on macOS
pytest
```

The tests run mlx-audio's randomly initialized tiny Music3 config and check the
node's staged generation matches `Model.generate` numerically, including
multi-chunk stitching.

## ComfyUI、节点和 Manager 的关系

- **ComfyUI** 是运行工作流的宿主程序；
- **本项目**是安装进 `custom_nodes` 的第三方插件，为宿主注册 4 个 MiniMax-H3 节点；
- **Comfy Registry** 保存插件的名称、版本、平台和安装归档；
- **ComfyUI Manager** 是插件商店/安装器，从 Registry 搜索、安装、升级和删除本项目。

节点不属于 ComfyUI 核心。未安装本插件时，ComfyUI 不认识 `MiniMaxH3Generate` 等节点；安装并重启后，ComfyUI 才会导入本项目的 `__init__.py` 并注册它们。

本项目已按 Registry ID `minimax-h3-mlx` 准备发布元数据。Registry 中现有的 `minimax-h3` 是另一个 CUDA/PyTorch 项目，不是本 MLX 节点。

## 节点

- **Load MiniMax H3 (MLX)**：加载 BF16 或 Argus INT8 模型、Text Encoder 与 Turbo LoRA；支持上游分阶段低内存模式。
- **Generate MiniMax H3 Video + Audio**：文生视频/音频，返回内存友好的 `MINIMAX_H3_RESULT`。
- **MiniMax H3 Result to Images + Audio**：按需转成 ComfyUI `IMAGE` 批次和 `AUDIO`。不需要逐帧后处理时可不接此节点，避免额外的 float32 帧内存。
- **Save MiniMax H3 MP4**：直接编码 H.264 + AAC MP4，并在节点内预览。

基本连接：

```text
Load MiniMax H3 (MLX) -> Generate MiniMax H3 Video + Audio -> Save MiniMax H3 MP4
                                                       \-> Result to Images + Audio（可选）
```

## 环境要求

- Apple Silicon Mac（arm64 macOS）
- Python 3.11 或 3.12
- 较新的 ComfyUI（需要 `AUDIO` 类型时请更新 ComfyUI）
- `ffmpeg`
- 推荐至少 24GB 统一内存和足够的 SSD 空间

BF16 Transformer 约 62GiB；INT8 Transformer 的有效文件约 37.8GB，另外还需要官方 tokenizer、processor、VAE 与 Text Encoder。

## 安装

### 方法 A：ComfyUI Manager（Registry 发布后推荐）

打开 Manager，搜索 **MiniMax H3 MLX for Mac** 或 Registry ID `minimax-h3-mlx`，点击安装并重启 ComfyUI。Manager 会从 Comfy Registry CDN 下载版本化 ZIP 并安装 `pyproject.toml`/`requirements.txt` 中声明的依赖；终端用户不需要执行 `git clone`。

### 方法 B：手动 ZIP（当前可用）

本节点 ZIP 已内置固定版本的 `minimax_h3_mlx` Python 代码。用户不需要安装 Git、克隆本节点仓库或单独克隆上游仓库。

1. 解压 `ComfyUI-MiniMax-H3-MLX.zip`；
2. 把解压得到的整个 `ComfyUI-MiniMax-H3-MLX` 文件夹放入 `ComfyUI/custom_nodes/`；
3. 使用启动 ComfyUI 的同一个 Python 安装 PyPI 依赖：

```bash
cd ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-MLX
/path/to/comfyui-python -m pip install -r requirements.txt
brew install ffmpeg
```

最终必须存在：

```text
ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-MLX/__init__.py
ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-MLX/minimax_h3_mlx/
```

安装后重启 ComfyUI。此安装过程不会访问 GitHub，但 `pip` 仍需访问 PyPI；模型需要从 Hugging Face 单独下载。

## 模型目录

建议目录结构：

```text
ComfyUI/models/minimax_h3/
├── MiniMax-H3/
│   └── FL2VA/
│       ├── model_index.json
│       ├── tokenizer/
│       ├── processor/
│       ├── text_encoder/
│       ├── transformer/       # 使用官方 BF16 时存在
│       ├── video_vae/
│       └── audio_vae/
├── MiniMax-H3-MLX-Argus-Calibrated-INT8/
├── MiniMax-H3-MLX-TextEncoder-4bit/
└── MiniMax-H3-Turbo-v4-step600-EMA-MLX/
```

首次下载前，需要在 Hugging Face 阅读并接受 [MiniMax-H3 License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)，然后登录：

```bash
/path/to/comfyui-python -m pip install -U huggingface_hub
hf auth login
```

### 下载官方基础组件

```bash
cd ComfyUI
mkdir -p models/minimax_h3

hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/model_index.json" \
            "FL2VA/tokenizer/**" \
            "FL2VA/processor/**" \
            "FL2VA/text_encoder/**" \
            "FL2VA/video_vae/**" \
            "FL2VA/audio_vae/**" \
  --local-dir models/minimax_h3/MiniMax-H3
```

### 路线 A：官方 BF16 Transformer + LightX2V Turbo

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/transformer/**" \
  --local-dir models/minimax_h3/MiniMax-H3

hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --revision e6346777701aa2b64d42ed058cdd71ae00e7cd52 \
  --local-dir models/minimax_h3/Minimax-h3-Turbo-v1.0-4step-768p-bf16
```

Loader 推荐值：

- `checkpoint_dir`：`.../models/minimax_h3/MiniMax-H3/FL2VA`
- `transformer_dir`：留空（使用 `FL2VA/transformer`）
- `turbo_lora_path`：上面下载的 `.safetensors`
- `low_memory=true`
- `stream_blocks=true`
- `stream_block_group_size=2`（24GB M4 Pro 上游实测配置）
- `sigma_shift_video=6`
- `sigma_shift_audio=3`

Generate 推荐先用 `steps=5`（对应 4 NFE）做 64×64、1 秒冒烟测试，再提高分辨率。

### 路线 B：Argus INT8 + 原生 MLX Turbo

```bash
hf download water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --revision 70505b09c80e298e684d798d8e0f946937dcfad4 \
  --local-dir models/minimax_h3/MiniMax-H3-MLX-Argus-Calibrated-INT8

hf download water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --revision 9771f9c606a50671bb94ee57191461602f02d1fc \
  --local-dir models/minimax_h3/MiniMax-H3-Turbo-v4-step600-EMA-MLX
```

4-bit Text Encoder 是可选项；留空即可使用官方 BF16 Text Encoder 的流式路径。如果需要 4-bit，可直接使用节点 ZIP 内置的转换工具，不需要克隆上游：

```bash
cd ComfyUI
/path/to/comfyui-python \
  custom_nodes/ComfyUI-MiniMax-H3-MLX/tools/quantize_text_encoder.py \
  --source models/minimax_h3/MiniMax-H3/FL2VA/text_encoder \
  --output models/minimax_h3/MiniMax-H3-MLX-TextEncoder-4bit \
  --bits 4 --group-size 64 --num-layers 50
```

Loader 配置：

- `checkpoint_dir`：官方 `MiniMax-H3/FL2VA`
- `transformer_dir`：`MiniMax-H3-MLX-Argus-Calibrated-INT8`
- `text_encoder_dir`：`MiniMax-H3-MLX-TextEncoder-4bit`（也可留空使用官方 BF16）
- `turbo_lora_path`：`MiniMax-H3-Turbo-v4-step600-EMA-MLX`
- `turbo_lora_alpha=0`（原生 MLX Turbo 已记录 alpha，不要手工覆盖）
- `low_memory=true`、`stream_blocks=true`

上游已实测 768×448、5 秒、`steps=9`（8 NFE）约 23 分钟，设备为 24GB M4 Pro。实际速度取决于芯片、提示词、SSD、温度和后台负载。

## 参数说明

- `steps` 是 **sigma grid points**，实际 DiT forward/NFE 数为 `steps - 1`。
- 宽高必须是 32 的倍数。H3 发布目标为 768 像素短边；更小尺寸适合检查节点连线，但属于分布外分辨率。
- `block_cache=false` 保留完整计算；开启后是近似残差缓存，可能改变结果。
- `memory_pressure_guard` 是上游实验选项，默认关闭。
- `turbo_lora_alpha=0` 表示不覆盖，读取 safetensors metadata 或原生 MLX adapter 配置。
- `sigma_shift_video/audio=0` 表示使用模型默认值。
- `Result to Images + Audio` 的 `frames=all` 会创建 float32 ComfyUI 图像批次。1344×768、约 124 帧会再占约 1.5GB 内存；只保存 MP4 时不要连接此转换节点。

## 图生视频限制

节点提供可选 `start_image` / `end_image` 接口，但上游当前明确限制：

- `low_memory=true` 只支持文生视频；
- 图像条件需要 Loader 设置 `low_memory=false` 且 `load_vision=true`；
- 对 24GB 设备通常不实用，是否可运行取决于所用量化模型和可用统一内存。

## 注意事项

- ComfyUI、PyTorch 与 MLX 共用统一内存。生成时请避免同时驻留其他大型 ComfyUI 模型。
- 1344×768、5 秒 BF16 在上游 24GB M4 Pro 实测接近 48 分钟，不是实时生成。
- MP4 保存依赖 ffmpeg；节点内可填写绝对路径，留空则搜索 `PATH`。
- 本节点代码使用 Apache-2.0；模型权重仍受各自上游许可证约束。

## 上游

- MLX 实现：<https://github.com/Argus-AiTeam/minimax-h3-mac>
- MiniMax-H3：<https://huggingface.co/MiniMaxAI/MiniMax-H3>
