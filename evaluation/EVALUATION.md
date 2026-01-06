# Evaluation

The inference time of LightGlue-ONNX is compared to that of the original PyTorch implementation with adaptive configuration and FlashAttention.

## Methods

Following the implementation details of the [LightGlue paper](https://arxiv.org/abs/2306.13643), we report the inference time, or latency, of only the LightGlue matcher; that is, the time taken for feature extraction, postprocessing, copying data between the host & device, or finding inliers (e.g., CONSAC/MAGSAC) is not measured. The average inference time is defined as the median over all samples in the [MegaDepth](https://arxiv.org/abs/1804.00607) test dataset. We use the data provided by [LoFTR](https://arxiv.org/abs/2104.00680) [here](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md) - a total of 403 image pairs.

Each image is resized such that its longer side is 1024 before being fed into the feature extractor. The average inference time of the LightGlue matcher is then measured for different numbers of keypoints: 512, 1024, 2048, and 4096. The [SuperPoint](http://arxiv.org/abs/1712.07629) extractor is used.

All experiments are conducted on an i9-12900HX CPU and RTX4080 12GB GPU with `CUDA==11.8.1`, `TensorRT==8.6.1`, `torch==2.1.0`, and `onnxruntime==1.16.0`.

## Results

The measured latencies are plotted in the figure below as image pairs per second.

<p align="center"><a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/evaluation/EVALUATION.md"><img src="../assets/latency.png" alt="Latency Comparison" width=100%></a>

<table align="center"><thead><tr><th>Number of Keypoints</th><th>512</th><th>1024</th><th>2048</th><th>4096</th></tr><tr><th>Model</th><th colspan="4">Latency (ms)</th></tr></thead><tbody><tr><td>PyTorch (Adaptive)</td><td>12.81</td><td>13.65</td><td>16.49</td><td>24.35</td></tr><tr><td>ORT Fused FP32</td><td>9.52</td><td>14.90</td><td>36.21</td><td>97.37</td></tr><tr><td>ORT Fused FP16</td><td>7.48</td><td>9.06</td><td>12.99</td><td>28.97</td></tr><tr><td>TensorRT FP16</td><td>7.11</td><td>7.56</td><td>10.81</td><td>24.46</td></tr></tbody></table>

In general, the fused ORT models can match the speed of the adaptive PyTorch model despite being non-adaptive (going through all attention layers). The PyTorch model provides more consistent latencies across the board, while the fused ORT models become slower at higher keypoint numbers due to a bottleneck in the `NonZero` operator. On the other hand, the TensorRT Execution Provider can reach very low latencies, but it is also inconsistent and unpredictable.
