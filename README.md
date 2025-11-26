# FLUX.1-dev Inference Optimization

> Export FLUX.1-dev model to ONNX format and TensorRT Engine for optimized inference

[中文版](README-zh.md) | English

## ✨ Features

- 🔄 Export FLUX.1-dev Transformer to ONNX format
- ⚡ TensorRT Engine acceleration for faster inference
- 🎨 Complete image generation pipeline (text to image)
- 📊 Multiple precision support (FP32, FP16, BF16)
- 🖼️ 1024x1024 high-quality image generation

## 📋 TODO

- [ ] INT8 quantization support (TensorRT)
- [ ] FP8 precision support (requires Hopper GPU)
- [ ] Dynamic resolution support
- [ ] Batch inference support
- [ ] Multi-GPU inference
- [ ] CUDA Graph optimization
- [ ] Streaming output support

## 📁 Project Structure

```
flux-inference/
├── base_model.py       # PyTorch baseline inference
├── onnx_model.py       # ONNX export tool
├── onnx_infer.py       # ONNX inference pipeline
├── tensorrt_model.py   # TensorRT engine builder
├── tensorrt_infer.py   # TensorRT inference pipeline
├── models/             # Exported models directory
│   ├── flux_transformer_{precision}/  # ONNX model
│   └── flux_transformer_{precision}.engine  # TensorRT engine
├── temp/               # Temporary files during export
├── results/            # Generated images
│   ├── baseline_flux_{precision}.png
│   ├── onnx_output_{precision}.png
│   └── tensorrt_output_{precision}.png
├── requirements.txt
├── README.md           # Chinese documentation
└── README_EN.md        # English documentation (this file)
```

## 🖼️ Generated Results

### Comparison

| Baseline (BF16) | ONNX (FP16) | ONNX (BF16) |
|:---:|:---:|:---:|
| ![Baseline BF16](results/baseline_flux_bf16.png) | ![ONNX FP16](results/onnx_output_fp16.png) | ![ONNX BF16](results/onnx_output_bf16.png) |

| TensorRT (FP16) | TensorRT (BF16) |
|:---:|:---:|
| ![TensorRT FP16](results/tensorrt_output_fp16.png) | ![TensorRT BF16](results/tensorrt_output_bf16.png) |

**Prompt**: `"A cat holding a sign that says hello world"`

## 🔧 Requirements

### Hardware Requirements
- NVIDIA GPU (tested on A800-SXM4-80GB)
- CUDA 12.0+
- Recommended 40GB+ VRAM

### Software Requirements
```bash
pip install -r requirements.txt
```

Main dependencies:
- torch >= 2.8.0
- diffusers >= 0.35.0
- transformers >= 4.50.0
- onnxruntime-gpu >= 1.20.0
- tensorrt >= 10.0.0 (separate installation required)

## 🚀 Quick Start

### 1. Baseline Inference (PyTorch)

```bash
# Run baseline inference
python base_model.py

# Output: results/baseline_flux_{PRECISION}.png
```

### 2. Export ONNX Model

```bash
# Export FLUX Transformer to ONNX format
python onnx_model.py

# Model saved to: models/flux_transformer_{PRECISION}/
```

### 3. ONNX Inference

```bash
# Run full image generation pipeline
python onnx_infer.py

# Output: results/onnx_output_{PRECISION}.png
```

### 4. Build TensorRT Engine

```bash
# Build TensorRT engine (first build takes 5-10 minutes)
python tensorrt_model.py

# Engine saved to: models/flux_transformer_{PRECISION}.engine
```

### 5. TensorRT Inference

```bash
# Run full image generation pipeline
python tensorrt_infer.py

# Output: results/tensorrt_output_{PRECISION}.png
```

## ⚙️ Configuration

Edit the configuration in the `main()` function of each file:

### Precision Options

| Precision | Description | VRAM Usage | Speed |
|-----------|-------------|------------|-------|
| `fp32` | Full precision | Highest | Slowest |
| `fp16` | Half precision | Medium | Fast |
| `bf16` | Brain floating point | Medium | Fast |

### Important Parameters

```python
# Model path
repo_root = "/path/to/FLUX.1-dev"

# Precision (fp32, fp16, bf16)
PRECISION = "bf16"

# Output resolution (fixed 1024x1024)
height, width = 1024, 1024

# Number of inference steps
num_inference_steps = 28
```

## 🔬 Performance Testing

### Benchmark Mode

```bash
# Modify run_full_pipeline() call
run_full_pipeline(repo_root, PRECISION, mode="benchmark")
```

### Check Mode

Generates a simple test image to verify the pipeline:

```bash
# Modify run_full_pipeline() call
run_full_pipeline(repo_root, PRECISION, mode="check")
```

## 📝 Notes

### About FP8/INT8 Support

- **FP8**: Requires Hopper GPU (H100+), TensorRT 9.0+, CUDA 12.0+
- **INT8**: Requires calibration dataset

Currently, only FP32/FP16/BF16 are implemented. FP8/INT8 support is planned for future releases.

### About Dynamic Resolution

Current implementation uses fixed dimensions (1024x1024). Dynamic resolution support is planned for future releases.

## 🐛 Troubleshooting

### 1. TensorRT External Weights Not Found

This is handled automatically by `tensorrt_model.py` which changes the working directory during engine build.

### 2. OOM (Out of Memory) Error

ONNX export requires a lot of VRAM. Make sure you have at least 40GB of VRAM available.

### 3. ONNX GPU Execution Issues

Run the check script first:

```bash
# Check ONNX GPU support
python onnx_infer.py check
```

### 4. Long TensorRT Build Time

First build takes 5-10 minutes, this is normal. The engine will be cached and subsequent loads are fast (~20s).

### 5. Insufficient VRAM

- Ensure you have enough GPU VRAM (40GB+ recommended)
- Close other GPU applications
- For TensorRT, adjust the `max_workspace_size` parameter

## 🔗 References

- [FLUX.1 Official Repo](https://github.com/black-forest-labs/flux)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## 📄 License

This project code is licensed under the MIT License.

FLUX.1-dev model usage is subject to its original license.

## 🙏 Acknowledgments

- **[Black Forest Labs](https://blackforestlabs.ai/)** - For open-sourcing FLUX.1 model
- **[Hugging Face](https://huggingface.co/)** - For Diffusers library and model hosting
- **[NVIDIA](https://nvidia.com/)** - For TensorRT and CUDA ecosystem
- **[GitHub Copilot](https://github.com/features/copilot)** - AI programming assistant that significantly improved development efficiency 🤖✨

---

*Made with ❤️ and GitHub Copilot*
