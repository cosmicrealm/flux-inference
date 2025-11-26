import torch
import os
import time
from diffusers import FluxPipeline

# 模型路径配置
REPO_ROOT = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-dev"

def run_baseline(
    prompt="A futuristic cyberpunk city with neon lights, ultra detailed, 8k resolution",
    save_name="baseline_flux.png",
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    seed=42
):
    """
    运行 FLUX.1-dev 基础模型推理
    
    Args:
        prompt: 文本提示词
        save_name: 保存图片的路径
        height: 图片高度
        width: 图片宽度
        num_inference_steps: 推理步数
        guidance_scale: 引导系数
        seed: 随机种子
    """
    print("=" * 80)
    print("FLUX.1-dev Baseline Inference")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print(f"Output: {save_name}")
    print("-" * 80)
    
    # 设置随机种子以保证可复现性
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # 1. 加载模型 (使用 bfloat16 以节省显存)
    print("Loading FLUX.1-dev pipeline...")
    start_time = time.time()
    
    pipe = FluxPipeline.from_pretrained(
        REPO_ROOT,
        torch_dtype=torch.bfloat16
    )
    
    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Warning: CUDA not available, using CPU (will be very slow)")
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    print("-" * 80)
    
    # 2. 推理
    print("Generating image...")
    start_time = time.time()
    
    # 清空 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed) if seed else None
    ).images[0]
    
    inference_time = time.time() - start_time
    
    # 3. 保存结果
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    image.save(save_name)
    
    # 4. 打印统计信息
    print(f"Image generated in {inference_time:.2f}s")
    print(f"Time per step: {inference_time / num_inference_steps:.3f}s")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU Memory: {peak_memory:.2f} GB")
    
    print(f"Image saved to: {save_name}")
    print("=" * 80)
    
    return image, {
        "load_time": load_time,
        "inference_time": inference_time,
        "time_per_step": inference_time / num_inference_steps,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    }


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    PRECISION = "bf16"  # baseline 使用 bfloat16
    # =================================================
    
    print("\n[Test 1] Standard 1024x1024 generation")
    save_name = f"results/baseline_flux_{PRECISION}.png"
    prompt = "A masterpiece photo of a beautiful sunset over rugged mountains, with dramatic, fiery clouds filling the sky. In the foreground, a golden retriever and a fluffy calico cat sit side-by-side on a rocky outcrop, looking out at the view. Cinematic lighting casts long shadows and warm golden light on their fur. 8k resolution, detailed textures."
    image, stats = run_baseline(
        prompt=prompt,
        save_name=save_name,
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=42
    )
    
    print("\n✅ All baseline tests completed!")