<div align="right"> <a href="https://github.com/fabio-sim/LightGlue-ONNX">English</a> | 简体中文 | <a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/docs/README.ja.md">日本語</a></div>

[![ONNX](https://img.shields.io/badge/ONNX-grey)](https://onnx.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900)](https://developer.nvidia.com/tensorrt)
[![GitHub Repo stars](https://img.shields.io/github/stars/fabio-sim/LightGlue-ONNX)](https://github.com/fabio-sim/LightGlue-ONNX/stargazers)
[![GitHub all releases](https://img.shields.io/github/downloads/fabio-sim/LightGlue-ONNX/total)](https://github.com/fabio-sim/LightGlue-ONNX/releases)
[![Blog](https://img.shields.io/badge/Blog-blue)](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/)

# LightGlue ONNX

兼容 Open Neural Network Exchange (ONNX) 的 [LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue) 实现。ONNX 模型格式支持跨平台互操作性，支持多种执行提供程序，并消除了诸如 PyTorch 之类的 Python 特定依赖。支持 TensorRT 和 OpenVINO。

> ✨ ***更新内容***：支持端到端并行动态批量大小。阅读更多内容，请查看这篇[博客文章](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/)。

<p align="center"><a href="https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/"><img src="../assets/inference-comparison-speedup.svg" alt="延迟对比" width=90%></a><br><em>⏱️ 推理时间对比</em></p>

<p align="center"><a href="https://arxiv.org/abs/2306.13643"><img src="../assets/easy_hard.jpg" alt="LightGlue 图示" width=80%></a></p>

<details>
<summary>更新日志</summary>

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

导出支持（包含 PyTorch + ONNX）：

```shell
uv sync --group export
```

TensorRT CLI 支持：

```shell
uv sync --group trt
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

使用 `--help` 参数可以查看每个命令的可用选项。CLI 将导出完整的提取器-匹配器管道，因此你不必担心中间步骤的协调。

### GPU 前提条件
ONNX Runtime 的 CUDA/TensorRT 执行提供程序需要兼容的 CUDA/cuDNN。如果你通过 PyPI 安装 CUDA/TensorRT 运行时（例如 `onnxruntime-gpu[cuda,cudnn]`、`tensorrt`），可能需要把 venv 里的路径加入 `LD_LIBRARY_PATH`，确保 Polygraphy/TRT EP 能找到 `libcudart.so` 和 `libnvinfer.so`：

```shell
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/tensorrt_libs:$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
```

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
