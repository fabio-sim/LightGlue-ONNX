<div align="right"> <a href="https://github.com/fabio-sim/LightGlue-ONNX">English</a> | 简体中文 | <a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/docs/README.ja.md">日本語</a></div>

[![ONNX](https://img.shields.io/badge/ONNX-grey)](https://onnx.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900)](https://developer.nvidia.com/tensorrt)
[![GitHub Repo stars](https://img.shields.io/github/stars/fabio-sim/LightGlue-ONNX)](https://github.com/fabio-sim/LightGlue-ONNX/stargazers)
[![GitHub all releases](https://img.shields.io/github/downloads/fabio-sim/LightGlue-ONNX/total)](https://github.com/fabio-sim/LightGlue-ONNX/releases)
[![Blog](https://img.shields.io/badge/Blog-blue)](https://fabio-sim.github.io)

# LightGlue ONNX

兼容 Open Neural Network Exchange (ONNX) 的 [LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue) 实现。ONNX 模型格式支持跨平台互操作性，支持多种执行提供程序，并消除了诸如 PyTorch 之类的 Python 特定依赖。支持 TensorRT 和 OpenVINO。[详细介绍](https://fabio-sim.github.io)。

> ✨ ***更新内容***：优化后的 RaCo-ALIKED-LightGlue+。阅读更多内容，请查看这篇[博客文章](https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/)。

<p align="center"><a href="https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/"><img src="../assets/marunouchi-animation.webp" alt="RaCo-ALIKED-LightGlue+" width=90%></a></p>

**2026年7月20日**：添加 RaCo-ALIKED-LightGlue+ 和基准测试。

<details>
<summary>更新日志</summary>

- **2026年1月19日**：添加 FP8 量化工作流说明（ModelOpt Q/DQ 导出与 TensorRT 用法）。[博客文章](https://fabio-sim.github.io/blog/fp8-quantized-lightglue-tensorrt-nvidia-model-optimizer/)
- **2026年1月9日**：用现代化 uv 刷新 CLI 体验，统一 `lightglue-onnx` 工作流，清理过时栈，同时更新依赖并补充 TensorRT/形状推断指引。
- **2024年7月17日**：支持端到端并行动态批量大小。重构脚本用户体验。添加[博客文章](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/)。
- **2023年11月2日**：引入 TopK-trick 来优化 ArgMax，提升约 30% 的速度。
- **2023年10月4日**：通过 `onnxruntime>=1.16.0` 支持 FlashAttention-2 的 LightGlue ONNX 模型融合，长序列推理速度提升高达 80%。
- **2023年10月27日**：LightGlue-ONNX 被添加到 [Kornia](https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.OnnxLightGlue)！
- **2023年10月4日**：多头注意力融合优化。
- **2023年7月19日**：添加对 TensorRT 的支持。
- **2023年7月13日**：添加 Flash Attention 支持。
- **2023年7月11日**：添加混合精度支持。
- **2023年7月4日**：添加推理时间对比。
- **2023年7月1日**：添加 `max_num_keypoints` 提取器支持。
- **2023年6月30日**：添加对 DISK 提取器的支持。
- **2023年6月28日**：添加端到端 SuperPoint+LightGlue 导出及推理管道。
</details>

## ⭐ ONNX 导出与推理

我们提供了一个 [typer](https://github.com/tiangolo/typer) CLI `lightglue-onnx`，用于轻松导出 LightGlue 为 ONNX 模型，并使用 ONNX Runtime 进行推理。如果你希望立即尝试推理，可以从[此处](https://github.com/fabio-sim/LightGlue-ONNX/releases)下载已导出的 ONNX 模型。

## 📦 安装（uv）

仅推理（默认）：

```shell
uv sync
```

CPU 导出支持（包含 PyTorch、torchvision、ONNX 和 ONNX Script）：

```shell
uv sync --group export --extra torch-cpu
```

CUDA 导出与推理支持（Linux x86-64）：

```shell
uv sync --no-group cpu --group cuda --group export --extra torch-cuda
```

TensorRT CLI 支持：

```shell
uv sync --no-group cpu --group cuda --group export --group trt --extra torch-cuda
```

```shell
$ uv run lightglue-onnx --help

Usage: lightglue-onnx [OPTIONS] COMMAND [ARGS]...

LightGlue Dynamo CLI

╭─ 命令 ───────────────────────────────────────╮
│ export   导出 LightGlue 为 ONNX 模型。        │
│ infer    使用 LightGlue ONNX 模型进行推理。   │
| trtexec  使用 Polygraphy 进行纯 TensorRT 推理 |
╰──────────────────────────────────────────────╯
```

使用 `--help` 参数可以查看每个命令的可用选项。CLI 将导出完整的提取器-匹配器管道，因此你不必担心中间步骤的协调。默认情况下，推理会在 CUDA 可用时使用 CUDA；如果请求的提供程序无法加载，则回退到 CPU。

### GPU 前提条件
ONNX Runtime 的 CUDA 和 TensorRT 执行提供程序需要与你的平台兼容的 CUDA 和 cuDNN 版本。如果遇到提供程序加载错误，请根据 ONNX Runtime CUDA 提供程序文档检查 CUDA/cuDNN 配置。
如果通过 PyPI 安装 CUDA/TensorRT 运行时库（例如 `onnxruntime-gpu[cuda,cudnn]` 和 `tensorrt`），可能需要将虚拟环境路径加入 `LD_LIBRARY_PATH`，以便 Polygraphy 和 TensorRT EP 找到 `libcudart.so` 与 `libnvinfer.so`：

```shell
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/tensorrt_libs:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

CLI 会自动预加载 wheel 提供的这些库；不过，对于从 CLI 之外启动的第三方工具，该环境变量仍然有用。

## 📖 示例命令

<details>
<summary>🔥 ONNX 导出</summary>
<pre>
uv run lightglue-onnx export superpoint \
  --num-keypoints 1024 \
  -b 2 -h 1024 -w 1024 \
  -o weights/superpoint_lightglue_pipeline.onnx
</pre>
</details>

<details>
<summary>⚡ ONNX Runtime 推理 (CUDA)</summary>
<pre>
uv run lightglue-onnx infer \
  weights/superpoint_lightglue_pipeline.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 1024 -w 1024 \
  -d cuda
</pre>
</details>

<details>
<summary>🚀 ONNX Runtime 推理 (TensorRT)</summary>
<pre>
uv run lightglue-onnx infer \
  weights/superpoint_lightglue_pipeline.trt.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 1024 -w 1024 \
  -d tensorrt --fp16
</pre>
</details>

<details>
<summary>🧩 TensorRT 推理</summary>
<pre>
uv run lightglue-onnx trtexec \
  weights/superpoint_lightglue_pipeline.trt.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 1024 -w 1024 \
  --fp16
</pre>
</details>

<details>
<summary>🧪 量化（TensorRT FP8 Q/DQ）</summary>
<pre>
# 1) 导出静态形状 ONNX 模型
uv run lightglue-onnx export superpoint \
  --num-keypoints 1024 \
  -b 2 -h 1024 -w 1024 \
  -o weights/superpoint_lightglue_pipeline.static.onnx

# 2) 量化为 FP8（DQ-only 图）
uv run lightglue_dynamo/scripts/quantize.py \
  --input weights/superpoint_lightglue_pipeline.static.onnx \
  --output weights/superpoint_lightglue_pipeline.static.fp8.onnx \
  --extractor superpoint \
  --height 1024 --width 1024 \
  --quantize-mode fp8 \
  --dq-only \
  --simplify

# 3) 运行 TensorRT（显式量化模型）
uv run lightglue-onnx trtexec \
  weights/superpoint_lightglue_pipeline.static.fp8.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 1024 -w 1024 \
  --precision-constraints prefer --fp16
</pre>
</details>

<details>
<summary>🟣 ONNX Runtime 推理 (OpenVINO)</summary>
<pre>
uv run lightglue-onnx infer \
  weights/superpoint_lightglue_pipeline.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 512 -w 512 \
  -d openvino
</pre>
</details>

## 🌐 浏览器 WebGPU 演示

启动静态演示，然后打开 `http://localhost:8000`：

```shell
uvx static-http --directory web --port 8000 --localhost-only
```

## ⏱️ 推理加速与输出质量

与 `torch.compile()` 的基准对比（[查看详情](https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/)）：

<p align="center"><a href="https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/"><img src="../assets/postoptimization-speedup-heatmap.svg" alt="RaCo-ALIKED-LightGlue+ 加速" width=90%></a></p>

<p align="center"><a href="https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/"><img src="../assets/postoptimization-match-quality.svg" alt="RaCo-ALIKED-LightGlue+ 匹配质量" width=90%></a></p>

<p align="center"><a href="https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/"><img src="../assets/postoptimization-pareto-frontier.svg" alt="RaCo-ALIKED-LightGlue+ 帕累托前沿" width=90%></a></p>

## 致谢
如果您在论文或代码中使用了本仓库中的任何想法，请考虑引用 [LightGlue](https://arxiv.org/abs/2306.13643)、[SuperPoint](https://arxiv.org/abs/1712.07629) 和 [DISK](https://arxiv.org/abs/2006.13566) 的作者。此外，如果 ONNX 版本对您有所帮助，请考虑为此仓库加星。

```txt
@inproceedings{lindenberger23lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue}: Local Feature Matching at Light Speed},
  booktitle = {ArXiv PrePrint},
  year      = {2023}
}
```

```txt
@article{DBLP:journals/corr/abs-1712-07629,
  author       = {Daniel DeTone and
                  Tomasz Malisiewicz and
                  Andrew Rabinovich},
  title        = {SuperPoint: Self-Supervised Interest Point Detection and Description},
  journal      = {CoRR},
  volume       = {abs/1712.07629},
  year         = {2017},
  url          = {http://arxiv.org/abs/1712.07629},
  eprinttype    = {arXiv},
  eprint       = {1712.07629},
  timestamp    = {Mon, 13 Aug 2018 16:47:29 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1712-07629.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```txt
@article{DBLP:journals/corr/abs-2006-13566,
  author       = {Michal J. Tyszkiewicz and
                  Pascal Fua and
                  Eduard Trulls},
  title        = {{DISK:} Learning local features with policy gradient},
  journal      = {CoRR},
  volume       = {abs/2006.13566},
  year         = {2020},
  url          = {https://arxiv.org/abs/2006.13566},
  eprinttype    = {arXiv},
  eprint       = {2006.13566},
  timestamp    = {Wed, 01 Jul 2020 15:21:23 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2006-13566.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
