# FLUX.1-dev 推理优化项目

本项目提供了 FLUX.1-dev 模型的完整推理流程，包括多种推理后端和精度选项。

## ✨ 特性

- ✅ **PyTorch Baseline** - 原生 BF16 推理
- ✅ **ONNX Runtime** - 跨平台推理，支持 FP16/BF16
- ✅ **TensorRT** - NVIDIA GPU 极致加速，支持 FP16/BF16
- ✅ **完整图像生成** - 输入 prompt，输出图片
- ✅ **多精度支持** - FP32、FP16、BF16

## 📋 TODO

- [ ] INT8 量化支持 (TensorRT)
- [ ] FP8 精度支持 (需要 Hopper GPU)
- [ ] 动态分辨率支持
- [ ] Batch 推理优化
- [ ] 多 GPU 并行推理
- [ ] Text Encoder / VAE 的 TensorRT 加速
- [ ] CUDA Graph 优化

## 🖼️ 生成结果

使用相同的 prompt 和 seed，不同推理后端的生成效果：

**Prompt**: *"A masterpiece photo of a beautiful sunset over rugged mountains, with dramatic, fiery clouds filling the sky. In the foreground, a golden retriever and a fluffy calico cat sit side-by-side on a rocky outcrop, looking out at the view."*

| Baseline (BF16) | ONNX (FP16) | ONNX (BF16) |
|:---------------:|:-----------:|:-----------:|
| ![baseline](results/baseline_flux_bf16.png) | ![onnx_fp16](results/onnx_output_fp16.png) | ![onnx_bf16](results/onnx_output_bf16.png) |

| TensorRT (FP16) | TensorRT (BF16) |
|:---------------:|:---------------:|
| ![trt_fp16](results/tensorrt_output_fp16.png) | ![trt_bf16](results/tensorrt_output_bf16.png) |

## 📁 项目结构

```
flux-inference/
├── base_model.py           # PyTorch baseline 推理
├── onnx_model.py           # ONNX 模型导出
├── onnx_infer.py           # ONNX 完整图像生成
├── tensorrt_model.py       # TensorRT Engine 构建
├── tensorrt_infer.py       # TensorRT 完整图像生成
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
├── models/                 # 存放导出的模型
│   ├── flux_transformer_{precision}.onnx
│   ├── flux_transformer_{precision}_weights.bin
│   └── flux_transformer_{precision}.engine
└── results/                # 存放生成的图片
    ├── baseline_flux_bf16.png
    ├── onnx_output_{precision}.png
    └── tensorrt_output_{precision}.png
```

## 🔧 环境要求

### 硬件
- NVIDIA GPU (推荐 40GB+ 显存，如 A100/A800)
- CUDA 12.0+
- TensorRT 10.0+

### 软件
- Python 3.10+
- PyTorch 2.0+
- ONNX Runtime GPU 1.16+
- TensorRT 10.x

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
conda create -n flux python=3.10
conda activate flux

# 安装 PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio

# 安装项目依赖
pip install -r requirements.txt

# 安装 ONNX Runtime GPU 版本
pip uninstall onnxruntime -y
pip install onnxruntime-gpu

# TensorRT 通常随 CUDA 一起安装，或从 NVIDIA 官网下载
```

### 2. 下载 FLUX.1-dev 模型

```bash
huggingface-cli login
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir /path/to/models/FLUX.1-dev
```

### 3. 配置模型路径

修改各文件 `__main__` 中的 `REPO_ROOT` 变量：

```python
REPO_ROOT = "/path/to/your/FLUX.1-dev"
```

## 📖 使用方法

### Baseline PyTorch 推理

```bash
python base_model.py
```

输出: `results/baseline_flux_bf16.png`

### ONNX 导出和推理

```bash
# Step 1: 导出 ONNX 模型 (首次需要，约 5 分钟)
python onnx_model.py

# Step 2: 使用 ONNX 生成图像
python onnx_infer.py

# 或者只运行基准测试
python onnx_infer.py benchmark

# 检查 GPU 支持
python onnx_infer.py check
```

输出: `results/onnx_output_{precision}.png`

### TensorRT 加速推理

```bash
# Step 1: 构建 TensorRT Engine (首次需要，约 5-10 分钟)
python tensorrt_model.py

# Step 2: 使用 TensorRT 生成图像
python tensorrt_infer.py

# 或者只运行基准测试
python tensorrt_infer.py benchmark
```

输出: `results/tensorrt_output_{precision}.png`

## ⚙️ 精度配置

在各文件的 `__main__` 函数中修改 `PRECISION` 变量：

```python
PRECISION = "bf16"  # 可选: "fp32", "fp16", "bf16"
```

### 精度说明

| 精度 | ONNX | TensorRT | 说明 |
|------|:----:|:--------:|------|
| FP32 | ✅ | ❌ | 标准精度，显存占用大 |
| FP16 | ✅ | ✅ | 推荐，速度快 |
| BF16 | ✅* | ✅ | 导出时转为 FP16 |
| INT8 | ⏳ | ⏳ | TODO: 需要校准数据 |
| FP8 | ❌ | ⏳ | TODO: 需要 Hopper GPU |

*BF16 导出时会转换为 FP16，因为 ONNX Runtime 对 BF16 支持有限

## 📊 性能对比

基于 NVIDIA A800 (80GB) 的测试结果（1024x1024，28 steps）：

| 方法 | Transformer 推理时间 | 完整生成时间 | 显存占用 |
|------|---------------------|-------------|---------|
| PyTorch (BF16) | ~350ms/step | ~12s | ~45GB |
| ONNX Runtime (FP16) | ~300ms/step | ~10s | ~40GB |
| TensorRT (FP16) | ~180ms/step | ~7s | ~35GB |

*实际性能取决于具体硬件和配置*

## ❓ 常见问题

### 1. ONNX 导出失败：`rms_norm` 不支持

已在 `onnx_model.py` 中注册了自定义的 `rms_norm` 符号化函数。

### 2. TensorRT 找不到外部权重

确保 ONNX 文件和 `*_weights.bin` 在同一目录下。

### 3. ONNX Runtime 没有使用 GPU

```bash
# 确保安装的是 GPU 版本
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu

# 运行检查
python onnx_infer.py check
```

### 4. TensorRT 构建时间过长

首次构建需要 5-10 分钟，这是正常的。Engine 会被缓存，后续加载很快（~20s）。

### 5. 显存不足 (OOM)

- 确保有足够的 GPU 显存 (推荐 40GB+)
- 关闭其他 GPU 程序
- 对于 TensorRT，调整 `max_workspace_size` 参数

## 🔗 参考资源

- [FLUX.1 Official Repo](https://github.com/black-forest-labs/flux)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## 📄 License

本项目代码遵循 MIT License。

FLUX.1-dev 模型的使用需遵循其原始许可证。

## 🙏 致谢

- **[Black Forest Labs](https://blackforestlabs.ai/)** - 开源 FLUX.1 模型
- **[Hugging Face](https://huggingface.co/)** - Diffusers 库和模型托管
- **[NVIDIA](https://nvidia.com/)** - TensorRT 和 CUDA 生态
- **[GitHub Copilot](https://github.com/features/copilot)** - AI 编程助手，大幅提升了开发效率 🤖✨

---

*Made with ❤️ and GitHub Copilot*
