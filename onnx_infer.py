"""
FLUX.1-dev ONNX Inference Script

使用 ONNX Runtime 进行完整的图像生成流程。

注意：需要安装 onnxruntime-gpu：
    pip uninstall onnxruntime
    pip install onnxruntime-gpu
"""

import onnxruntime as ort
import numpy as np
import torch
import time
import os
from PIL import Image

# 检查 ONNX Runtime GPU 支持
def check_onnx_gpu():
    """检查 ONNX Runtime GPU 支持状态"""
    print("=" * 60)
    print("ONNX Runtime Environment Check")
    print("=" * 60)
    print(f"ONNX Runtime version: {ort.__version__}")
    
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    
    if 'CUDAExecutionProvider' in available_providers:
        print("✅ CUDAExecutionProvider is available")
        return True
    else:
        print("❌ CUDAExecutionProvider NOT available!")
        print("\nTo enable GPU support, run:")
        print("  pip uninstall onnxruntime onnxruntime-gpu")
        print("  pip install onnxruntime-gpu")
        return False


class ONNXWrapper:
    """ONNX Runtime 推理封装（GPU 加速）"""
    
    def __init__(self, onnx_path, verbose=False, device_id=0, use_gpu=True):
        """
        初始化 ONNX Runtime 推理引擎
        
        Args:
            onnx_path: ONNX 模型文件路径
            verbose: 是否打印详细信息
            device_id: GPU 设备 ID
            use_gpu: 是否使用 GPU
        """
        self.verbose = verbose
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        
        if self.verbose:
            print(f"Loading ONNX model from {onnx_path}...")
        
        # 检查文件存在
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        
        # 配置 ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 启用内存优化
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        # 检查可用的 providers
        available_providers = ort.get_available_providers()
        print(f"Available ONNX providers: {available_providers}")
        
        # 配置 providers
        if use_gpu and 'CUDAExecutionProvider' in available_providers:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 60 * 1024 * 1024 * 1024,  # 60GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
            ]
            print(f"🚀 Configuring CUDA provider on GPU {device_id}")
        else:
            providers = ['CPUExecutionProvider']
            print("⚠️ Using CPU provider (GPU not available)")
        
        # 创建 Session
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # 验证实际使用的 provider
        active_providers = self.session.get_providers()
        print(f"Active providers: {active_providers}")
        
        self.use_cuda = 'CUDAExecutionProvider' in active_providers
        if self.use_cuda:
            print(f"✅ Running on GPU (CUDAExecutionProvider)")
        else:
            print(f"⚠️ Running on CPU")
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_info = {inp.name: inp for inp in self.session.get_inputs()}
        self.output_info = {out.name: out for out in self.session.get_outputs()}
        
        if self.verbose:
            print(f"\nInputs: {self.input_names}")
            print(f"Outputs: {self.output_names}")
            print("\nInput details:")
            for inp in self.session.get_inputs():
                print(f"  {inp.name}: {inp.shape} ({inp.type})")
    
    def _get_numpy_dtype(self, ort_type):
        """将 ONNX 类型字符串转换为 numpy dtype"""
        type_mapping = {
            'tensor(float16)': np.float16,
            'tensor(float)': np.float32,
            'tensor(int64)': np.int64,
            'tensor(int32)': np.int32,
            'tensor(bool)': np.bool_,
        }
        return type_mapping.get(ort_type, np.float32)
    
    def infer(self, input_tensors):
        """
        执行推理
        
        Args:
            input_tensors: 字典, {'input_name': numpy_array 或 torch.Tensor, ...}
        
        Returns:
            字典, {'output_name': torch.Tensor (CUDA), ...}
        """
        # 准备输入
        ort_inputs = {}
        for name in self.input_names:
            data = input_tensors[name]
            
            # 转换为 numpy
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            
            # 确保类型正确
            expected_dtype = self._get_numpy_dtype(self.input_info[name].type)
            if data.dtype != expected_dtype:
                data = data.astype(expected_dtype)
            
            ort_inputs[name] = data
        
        # 执行推理
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        
        # 转换输出为 CUDA torch tensor
        result = {}
        for name, output in zip(self.output_names, ort_outputs):
            result[name] = torch.from_numpy(output).to(self.device)
        
        return result


def run_full_pipeline(
    onnx_path,
    repo_root,
    prompt="A beautiful sunset over mountains, cinematic lighting, 8k resolution",
    negative_prompt="",
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    seed=42,
    save_path="results/onnx_output.png"
):
    """
    使用 ONNX Runtime 的完整 FLUX 图像生成流程
    
    手动组合各组件进行推理。
    """
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from tqdm import tqdm
    
    print("=" * 80)
    print("FLUX.1-dev ONNX Full Pipeline")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {num_inference_steps}")
    print(f"Guidance: {guidance_scale}")
    print(f"Seed: {seed}")
    print("-" * 80)
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # 1. 加载 ONNX 模型
    print("[1/6] Loading ONNX model...")
    start_time = time.time()
    onnx_wrapper = ONNXWrapper(onnx_path, verbose=True)
    print(f"ONNX model loaded in {time.time() - start_time:.2f}s")
    
    # 2. 加载 Text Encoders
    print("\n[2/6] Loading text encoders...")
    start_time = time.time()
    
    # CLIP Text Encoder
    tokenizer = CLIPTokenizer.from_pretrained(repo_root, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        repo_root, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    
    # T5 Text Encoder
    tokenizer_2 = T5TokenizerFast.from_pretrained(repo_root, subfolder="tokenizer_2")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        repo_root, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    
    print(f"Text encoders loaded in {time.time() - start_time:.2f}s")
    
    # 3. 加载 VAE
    print("[3/6] Loading VAE...")
    start_time = time.time()
    vae = AutoencoderKL.from_pretrained(
        repo_root, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    print(f"VAE loaded in {time.time() - start_time:.2f}s")
    
    # 4. 创建 Scheduler
    print("[4/6] Setting up scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo_root, subfolder="scheduler"
    )
    
    # FLUX.1-dev uses dynamic shifting
    image_seq_len = (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
    base_shift = 0.5
    max_shift = 1.15
    mu = base_shift + (max_shift - base_shift) * (image_seq_len - 256) / (4096 - 256)
    mu = max(base_shift, min(max_shift, mu))
    
    scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    
    # 5. 编码 prompt
    print("[5/6] Encoding prompt...")
    start_time = time.time()
    
    # CLIP encoding
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        prompt_embeds_clip = text_encoder(
            text_input_ids,
            output_hidden_states=False,
        )
        pooled_prompt_embeds = prompt_embeds_clip.pooler_output.to(dtype)  # [1, 768]
    
    # T5 encoding
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids_2 = text_inputs_2.input_ids.to(device)
    
    with torch.no_grad():
        prompt_embeds_t5 = text_encoder_2(
            text_input_ids_2,
            output_hidden_states=False,
        )[0].to(dtype)  # [1, 512, 4096]
    
    print(f"Prompt encoded in {time.time() - start_time:.2f}s")
    
    # 6. 准备 latents 和 IDs
    print("[6/6] Preparing latents...")
    
    # Latent dimensions
    latent_height = height // vae_scale_factor  # 1024 / 8 = 128
    latent_width = width // vae_scale_factor    # 1024 / 8 = 128
    num_channels_latents = 16  # FLUX uses 16 channels
    
    # 对于 FLUX，latents 被 pack 成 sequence
    packed_height = latent_height // 2  # 64
    packed_width = latent_width // 2    # 64
    seq_len = packed_height * packed_width  # 4096
    
    # 生成初始噪声
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, num_channels_latents, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    
    # Pack latents: [1, 16, 128, 128] -> [1, 4096, 64]
    latents_packed = latents.view(1, num_channels_latents, packed_height, 2, packed_width, 2)
    latents_packed = latents_packed.permute(0, 2, 4, 1, 3, 5).contiguous()
    latents_packed = latents_packed.view(1, seq_len, num_channels_latents * 4)  # [1, 4096, 64]
    
    # 准备 img_ids 和 txt_ids
    img_ids = torch.zeros(seq_len, 3, device=device, dtype=dtype)
    for i in range(seq_len):
        h_idx = i // packed_width
        w_idx = i % packed_width
        img_ids[i, 0] = 0
        img_ids[i, 1] = h_idx
        img_ids[i, 2] = w_idx
    
    text_seq_len = 512
    txt_ids = torch.zeros(text_seq_len, 3, device=device, dtype=dtype)
    
    # Guidance tensor
    guidance_tensor = torch.tensor([guidance_scale], device=device, dtype=dtype)
    
    # ==================== 开始去噪循环 ====================
    print(f"\nDenoising ({num_inference_steps} steps)...")
    generation_start = time.time()
    
    timesteps = scheduler.timesteps
    
    for i, t in enumerate(tqdm(timesteps, desc="Generating")):
        # 准备 timestep
        timestep = t.expand(1).to(dtype)
        
        # ONNX 推理
        inputs = {
            "hidden_states": latents_packed,
            "encoder_hidden_states": prompt_embeds_t5,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep / 1000.0,  # Normalize timestep
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance_tensor,
        }
        
        outputs = onnx_wrapper.infer(inputs)
        noise_pred = outputs["sample"]  # [1, 4096, 64]
        
        # Scheduler step
        latents_packed = scheduler.step(noise_pred, t, latents_packed, return_dict=False)[0]
    
    generation_time = time.time() - generation_start
    
    # ==================== VAE 解码 ====================
    print("\nDecoding latents with VAE...")
    
    # Unpack latents: [1, 4096, 64] -> [1, 16, 128, 128]
    latents_unpacked = latents_packed.view(1, packed_height, packed_width, num_channels_latents, 2, 2)
    latents_unpacked = latents_unpacked.permute(0, 3, 1, 4, 2, 5).contiguous()
    latents_unpacked = latents_unpacked.view(1, num_channels_latents, latent_height, latent_width)
    
    # VAE expects different scaling for FLUX
    latents_unpacked = (latents_unpacked / vae.config.scaling_factor) + vae.config.shift_factor
    
    with torch.no_grad():
        image = vae.decode(latents_unpacked, return_dict=False)[0]
    
    # Post-process image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype(np.uint8)[0]
    image = Image.fromarray(image)
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    
    print("=" * 80)
    print(f"✅ Image generated successfully!")
    print(f"Generation time: {generation_time:.2f}s ({generation_time/num_inference_steps*1000:.1f}ms/step)")
    print(f"Saved to: {save_path}")
    print("=" * 80)
    
    # 清理
    del text_encoder, text_encoder_2, vae
    torch.cuda.empty_cache()
    
    return image, {
        "generation_time": generation_time,
        "time_per_step": generation_time / num_inference_steps
    }


def run_benchmark(
    onnx_path,
    height=1024,
    width=1024,
    num_runs=5,
    seed=42
):
    """
    ONNX Transformer 性能基准测试（不生成完整图像）
    """
    print("=" * 80)
    print("ONNX Transformer Benchmark")
    print("=" * 80)
    print(f"Model: {onnx_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Runs: {num_runs}")
    print("-" * 80)
    
    # 加载模型
    print("Loading ONNX model...")
    onnx_wrapper = ONNXWrapper(onnx_path, verbose=True)
    
    # 准备输入
    torch.manual_seed(seed)
    seq_len = 4096
    text_seq_len = 512
    
    inputs = {
        "hidden_states": torch.randn(1, seq_len, 64, dtype=torch.float16, device="cuda"),
        "encoder_hidden_states": torch.randn(1, text_seq_len, 4096, dtype=torch.float16, device="cuda"),
        "pooled_projections": torch.randn(1, 768, dtype=torch.float16, device="cuda"),
        "timestep": torch.tensor([1.0], dtype=torch.float16, device="cuda"),
        "img_ids": torch.randn(seq_len, 3, dtype=torch.float16, device="cuda"),
        "txt_ids": torch.randn(text_seq_len, 3, dtype=torch.float16, device="cuda"),
        "guidance": torch.tensor([3.5], dtype=torch.float16, device="cuda")
    }
    
    # 预热
    print("\nWarming up...")
    for _ in range(3):
        _ = onnx_wrapper.infer(inputs)
    
    # 基准测试
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = onnx_wrapper.infer(inputs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("=" * 80)
    print("Results:")
    print(f"  Average: {avg_time * 1000:.2f} ms")
    print(f"  Min: {min_time * 1000:.2f} ms")
    print(f"  Max: {max_time * 1000:.2f} ms")
    print(f"  Throughput: {1.0 / avg_time:.2f} samples/sec")
    print(f"\nOutput shape: {list(outputs['sample'].shape)}")
    print("=" * 80)
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time
    }


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    REPO_ROOT = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-dev"
    OUTPUT_DIR = "models"
    MODEL_NAME = "flux_transformer"
    PRECISION = "bf16"  # 可选: "fp32", "fp16", "bf16"
    # =================================================
    
    # 首先检查 GPU 支持
    gpu_available = check_onnx_gpu()
    print()
    
    onnx_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{PRECISION}.onnx")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Error: ONNX model not found at {onnx_path}")
        print("Please run onnx_model.py first to export the model.")
        exit(1)
    
    onnx_size_gb = os.path.getsize(onnx_path) / (1024 ** 3)
    print(f"Found ONNX model: {onnx_path} ({onnx_size_gb:.2f} GB)")
    
    # 检查外部权重文件
    weights_path = onnx_path.replace(".onnx", "_weights.bin")
    if os.path.exists(weights_path):
        weights_size_gb = os.path.getsize(weights_path) / (1024 ** 3)
        print(f"Found external weights: {weights_path} ({weights_size_gb:.2f} GB)")
    
    if not gpu_available:
        print("\n⚠️ GPU not available for ONNX Runtime!")
        print("Install onnxruntime-gpu for GPU acceleration:")
        print("  pip uninstall onnxruntime")
        print("  pip install onnxruntime-gpu")
        
        response = input("\nContinue with CPU? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    # 选择运行模式
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    
    if mode == "benchmark":
        # 仅测试 Transformer 性能
        print("\n[Mode: Benchmark]")
        run_benchmark(onnx_path)
    elif mode == "check":
        # 只检查环境
        print("\n[Mode: Environment Check Only]")
    else:
        # 完整图像生成
        print("\n[Mode: Generate Image]")
        prompt = "A masterpiece photo of a beautiful sunset over rugged mountains, with dramatic, fiery clouds filling the sky. In the foreground, a golden retriever and a fluffy calico cat sit side-by-side on a rocky outcrop, looking out at the view. Cinematic lighting casts long shadows and warm golden light on their fur. 8k resolution, detailed textures."
        image, stats = run_full_pipeline(
            onnx_path=onnx_path,
            repo_root=REPO_ROOT,
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=3.5,
            seed=42,
            save_path=f"results/onnx_output_{PRECISION}.png"
        )
        
        print(f"\nGeneration completed in {stats['generation_time']:.2f}s")
