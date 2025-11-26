# FLUX.1-dev Inference Optimization

> Export FLUX.1-dev model to ONNX format and TensorRT Engine for optimized inference

[中文](README-zh.md) | English

This project provides a complete inference pipeline for FLUX.1-dev model, including multiple inference backends and precision options.

## ✨ Features

- ✅ **PyTorch Baseline** - Native BF16 inference
- ✅ **ONNX Runtime** - Cross-platform inference, supports FP16/BF16
- ✅ **TensorRT** - NVIDIA GPU extreme acceleration, supports FP16/BF16
- ✅ **Complete Image Generation** - Input prompt, output image
- ✅ **Multiple Precision Support** - FP32, FP16, BF16

## 📋 TODO

- [ ] INT8 quantization support (TensorRT)
- [ ] FP8 precision support (requires Hopper GPU)
- [ ] Dynamic resolution support
- [ ] Batch inference optimization
- [ ] Multi-GPU parallel inference
- [ ] Text Encoder / VAE TensorRT acceleration
- [ ] CUDA Graph optimization

## 🖼️ Generated Results

Using the same prompt and seed, generation results from different inference backends:

**Prompt**: *"A masterpiece photo of a beautiful sunset over rugged mountains, with dramatic, fiery clouds filling the sky. In the foreground, a golden retriever and a fluffy calico cat sit side-by-side on a rocky outcrop, looking out at the view."*

| Baseline (BF16) | ONNX (FP16) | ONNX (BF16) |
|:---------------:|:-----------:|:-----------:|
| ![baseline](results/baseline_flux_bf16.png) | ![onnx_fp16](results/onnx_output_fp16.png) | ![onnx_bf16](results/onnx_output_bf16.png) |

| TensorRT (FP16) | TensorRT (BF16) |
|:---------------:|:---------------:|
| ![trt_fp16](results/tensorrt_output_fp16.png) | ![trt_bf16](results/tensorrt_output_bf16.png) |

## 📁 Project Structure

```
flux-inference/
├── base_model.py           # PyTorch baseline inference
├── onnx_model.py           # ONNX model export
├── onnx_infer.py           # ONNX complete image generation
├── tensorrt_model.py       # TensorRT Engine build
├── tensorrt_infer.py       # TensorRT complete image generation
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Exported models directory
│   ├── flux_transformer_{precision}.onnx
│   ├── flux_transformer_{precision}_weights.bin
│   └── flux_transformer_{precision}.engine
└── results/                # Generated images directory
    ├── baseline_flux_bf16.png
    ├── onnx_output_{precision}.png
    └── tensorrt_output_{precision}.png
```

## 🔧 Requirements

### Hardware
- NVIDIA GPU (recommended 40GB+ VRAM, e.g., A100/A800)
- CUDA 12.0+
- TensorRT 10.0+

### Software
- Python 3.10+
- PyTorch 2.0+
- ONNX Runtime GPU 1.16+
- TensorRT 10.x

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
conda create -n flux python=3.10
conda activate flux

# Install PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio

# Install project dependencies
pip install -r requirements.txt

# Install ONNX Runtime GPU version
pip uninstall onnxruntime -y
pip install onnxruntime-gpu

# TensorRT is usually installed with CUDA, or download from NVIDIA website
```

### 2. Download FLUX.1-dev Model

```bash
huggingface-cli login
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir /path/to/models/FLUX.1-dev
```

### 3. Configure Model Path

Modify the `REPO_ROOT` variable in each file's `__main__`:

```python
REPO_ROOT = "/path/to/your/FLUX.1-dev"
```

## 📖 Usage

### Baseline PyTorch Inference

```bash
python base_model.py
```

Output: `results/baseline_flux_bf16.png`

### ONNX Export and Inference

```bash
# Step 1: Export ONNX model (required for first time, ~5 minutes)
python onnx_model.py

# Step 2: Generate image with ONNX
python onnx_infer.py

# Or run benchmark only
python onnx_infer.py benchmark

# Check GPU support
python onnx_infer.py check
```

Output: `results/onnx_output_{precision}.png`

### TensorRT Accelerated Inference

```bash
# Step 1: Build TensorRT Engine (required for first time, ~5-10 minutes)
python tensorrt_model.py

# Step 2: Generate image with TensorRT
python tensorrt_infer.py

# Or run benchmark only
python tensorrt_infer.py benchmark
```

Output: `results/tensorrt_output_{precision}.png`

## ⚙️ Precision Configuration

Modify the `PRECISION` variable in each file's `__main__` function:

```python
PRECISION = "bf16"  # Options: "fp32", "fp16", "bf16"
```

### Precision Details

| Precision | ONNX | TensorRT | Description |
|-----------|:----:|:--------:|-------------|
| FP32 | ✅ | ❌ | Standard precision, high VRAM usage |
| FP16 | ✅ | ✅ | Recommended, fast |
| BF16 | ✅* | ✅ | Converted to FP16 during export |
| INT8 | ⏳ | ⏳ | TODO: Requires calibration data |
| FP8 | ❌ | ⏳ | TODO: Requires Hopper GPU |

*BF16 is converted to FP16 during export due to limited ONNX Runtime BF16 support

## 📊 Performance Comparison

Test results based on NVIDIA A800 (80GB) (1024x1024, 28 steps):

| Method | Transformer Inference Time | Total Generation Time | VRAM Usage |
|--------|---------------------------|----------------------|------------|
| PyTorch (BF16) | ~350ms/step | ~12s | ~45GB |
| ONNX Runtime (FP16) | ~300ms/step | ~10s | ~40GB |
| TensorRT (FP16) | ~180ms/step | ~7s | ~35GB |

*Actual performance depends on specific hardware and configuration*

## ❓ FAQ

### 1. ONNX Export Failed: `rms_norm` Not Supported

A custom `rms_norm` symbolic function has been registered in `onnx_model.py`.

### 2. TensorRT Cannot Find External Weights

Make sure the ONNX file and `*_weights.bin` are in the same directory.

### 3. ONNX Runtime Not Using GPU

```bash
# Make sure GPU version is installed
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu

# Run check
python onnx_infer.py check
```

### 4. Long TensorRT Build Time

First build takes 5-10 minutes, this is normal. Engine will be cached, subsequent loads are fast (~20s).

### 5. Out of Memory (OOM)

- Ensure you have enough GPU VRAM (40GB+ recommended)
- Close other GPU programs
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
