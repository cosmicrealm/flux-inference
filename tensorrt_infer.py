import tensorrt as trt
import numpy as np
import torch
import time
import os
from PIL import Image
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipelineOutput

# 使用 torch 进行 CUDA 内存管理
torch.cuda.init()


class TRTWrapper:
    """TensorRT Engine 推理封装（使用 torch CUDA）"""
    
    def __init__(self, engine_path, verbose=False):
        """
        初始化 TensorRT 推理引擎
        
        Args:
            engine_path: TensorRT Engine 文件路径
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.device = torch.device("cuda:0")
        
        if self.verbose:
            print(f"Loading TensorRT engine from {engine_path}...")
        
        # 1. 创建 Logger 和 Runtime
        self.logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 2. 加载序列化的 Engine
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")
        
        # 3. 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 4. 创建 CUDA Stream
        self.stream = torch.cuda.Stream(device=self.device)
        
        # 5. 获取输入输出信息（TensorRT 10.x API）
        self.input_names = []
        self.output_names = []
        self.binding_dtypes = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            self.binding_dtypes[name] = dtype
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        if self.verbose:
            print(f"✅ Engine loaded successfully")
            print(f"Inputs: {self.input_names}")
            print(f"Outputs: {self.output_names}")
    
    def _trt_dtype_to_torch(self, trt_dtype):
        """将 TensorRT dtype 转换为 torch dtype"""
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)
    
    def infer(self, input_tensors):
        """
        执行推理
        
        Args:
            input_tensors: 字典, {'input_name': numpy_array 或 torch.Tensor, ...}
        
        Returns:
            字典, {'output_name': torch.Tensor, ...}
        """
        with torch.cuda.stream(self.stream):
            # 1. 设置输入 shape 和地址
            for name in self.input_names:
                data = input_tensors[name]
                
                # 转换为 torch tensor
                if isinstance(data, np.ndarray):
                    torch_dtype = self._trt_dtype_to_torch(self.binding_dtypes[name])
                    data = torch.from_numpy(data).to(dtype=torch_dtype, device=self.device)
                elif not data.is_cuda:
                    data = data.to(self.device)
                
                # 确保连续
                data = data.contiguous()
                input_tensors[name] = data
                
                # 设置输入 shape 和地址
                self.context.set_input_shape(name, tuple(data.shape))
                self.context.set_tensor_address(name, data.data_ptr())
            
            # 2. 分配输出缓冲区
            outputs = {}
            for name in self.output_names:
                shape = self.context.get_tensor_shape(name)
                torch_dtype = self._trt_dtype_to_torch(self.binding_dtypes[name])
                output_tensor = torch.empty(tuple(shape), dtype=torch_dtype, device=self.device)
                outputs[name] = output_tensor
                self.context.set_tensor_address(name, output_tensor.data_ptr())
            
            # 3. 执行推理
            self.context.execute_async_v3(self.stream.cuda_stream)
            
            # 4. 同步
            self.stream.synchronize()
        
        return outputs


class TRTTransformerWrapper(torch.nn.Module):
    """将 TensorRT Engine 包装为 PyTorch Module，用于替换 FluxPipeline 中的 Transformer"""
    
    def __init__(self, trt_wrapper, dtype=torch.float16):
        super().__init__()
        self.trt_wrapper = trt_wrapper
        self.dtype = dtype
        
        # Mock config with all required attributes from FLUX.1-dev transformer/config.json
        self.config = type('Config', (), {
            'in_channels': 64,
            'attention_head_dim': 128,
            'axes_dims_rope': [16, 56, 56],
            'guidance_embeds': True,
            'joint_attention_dim': 4096,
            'num_attention_heads': 24,
            'num_layers': 19,
            'num_single_layers': 38,
            'patch_size': 1,
            'pooled_projection_dim': 768,
        })()
    
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        pooled_projections=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
        **kwargs
    ):
        # 准备 TensorRT 输入
        inputs = {
            "hidden_states": hidden_states.to(self.dtype),
            "encoder_hidden_states": encoder_hidden_states.to(self.dtype),
            "pooled_projections": pooled_projections.to(self.dtype),
            "timestep": timestep.to(self.dtype),
            "img_ids": img_ids.to(self.dtype),
            "txt_ids": txt_ids.to(self.dtype),
            "guidance": guidance.to(self.dtype) if guidance is not None else torch.tensor([3.5], dtype=self.dtype, device=hidden_states.device),
        }
        
        # 执行 TensorRT 推理
        outputs = self.trt_wrapper.infer(inputs)
        
        # 返回结果
        sample = outputs["sample"]
        
        if return_dict:
            from diffusers.models.transformers.transformer_flux import FluxTransformer2DModelOutput
            return FluxTransformer2DModelOutput(sample=sample)
        return (sample,)


def run_full_pipeline(
    engine_path,
    repo_root,
    prompt="A beautiful sunset over mountains, cinematic lighting, 8k resolution",
    negative_prompt="",
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    seed=42,
    save_path="results/tensorrt_output.png"
):
    """
    使用 TensorRT 加速的完整 FLUX 图像生成流程
    
    不使用 FluxPipeline，而是手动组合各组件进行推理，
    以避免 diffusers 内部对 transformer 的特殊方法调用。
    """
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from tqdm import tqdm
    
    print("=" * 80)
    print("FLUX.1-dev TensorRT Full Pipeline")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {num_inference_steps}")
    print(f"Guidance: {guidance_scale}")
    print(f"Seed: {seed}")
    print("-" * 80)
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # 1. 加载 TensorRT 引擎
    print("[1/6] Loading TensorRT engine...")
    start_time = time.time()
    trt_wrapper = TRTWrapper(engine_path, verbose=False)
    print(f"Engine loaded in {time.time() - start_time:.2f}s")
    
    # 2. 加载 Text Encoders
    print("[2/6] Loading text encoders...")
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
    
    # FLUX.1-dev uses dynamic shifting, need to calculate mu based on image size
    # mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
    # For 1024x1024: seq_len = 4096, base_shift=0.5, max_shift=1.15
    image_seq_len = (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
    base_shift = 0.5
    max_shift = 1.15
    # Linear interpolation based on seq_len (256 to 4096 range)
    mu = base_shift + (max_shift - base_shift) * (image_seq_len - 256) / (4096 - 256)
    mu = max(base_shift, min(max_shift, mu))  # Clamp to valid range
    
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
    num_channels_latents = 16  # FLUX uses 16 channels in latent space
    
    # 对于 FLUX，latents 被 pack 成 sequence
    # packed latent shape: [batch, seq_len, channels] where seq_len = (H/2) * (W/2)
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
    # FLUX packing: reshape to [B, C, H/2, 2, W/2, 2] -> [B, H/2*W/2, C*4]
    latents_packed = latents.view(1, num_channels_latents, packed_height, 2, packed_width, 2)
    latents_packed = latents_packed.permute(0, 2, 4, 1, 3, 5).contiguous()
    latents_packed = latents_packed.view(1, seq_len, num_channels_latents * 4)  # [1, 4096, 64]
    
    # 准备 img_ids 和 txt_ids
    # img_ids: positional encoding for image patches [seq_len, 3]
    img_ids = torch.zeros(seq_len, 3, device=device, dtype=dtype)
    for i in range(seq_len):
        h_idx = i // packed_width
        w_idx = i % packed_width
        img_ids[i, 0] = 0  # batch index (always 0 for single image)
        img_ids[i, 1] = h_idx
        img_ids[i, 2] = w_idx
    
    # txt_ids: positional encoding for text [512, 3]
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
        
        # TensorRT 推理
        inputs = {
            "hidden_states": latents_packed,
            "encoder_hidden_states": prompt_embeds_t5,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep / 1000.0,  # Normalize timestep
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance_tensor,
        }
        
        outputs = trt_wrapper.infer(inputs)
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
    engine_path,
    height=1024,
    width=1024,
    num_runs=5,
    seed=42
):
    """
    TensorRT Transformer 性能基准测试（不生成完整图像）
    """
    print("=" * 80)
    print("TensorRT Transformer Benchmark")
    print("=" * 80)
    print(f"Engine: {engine_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Runs: {num_runs}")
    print("-" * 80)
    
    # 加载引擎
    print("Loading TensorRT engine...")
    trt_wrapper = TRTWrapper(engine_path, verbose=True)
    
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
    print("Warming up...")
    for _ in range(3):
        _ = trt_wrapper.infer(inputs)
    torch.cuda.synchronize()
    
    # 基准测试
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = trt_wrapper.infer(inputs)
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
    PRECISION = "bf16"  # 可选: "fp16", "bf16" (TensorRT 不支持 fp32 大模型)
    # =================================================
    
    engine_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{PRECISION}.engine")
    
    if not os.path.exists(engine_path):
        print(f"❌ Error: TensorRT engine not found at {engine_path}")
        print("Please run tensorrt_model.py first to build the engine.")
        exit(1)
    
    engine_size_gb = os.path.getsize(engine_path) / (1024 ** 3)
    print(f"Found TensorRT engine: {engine_path} ({engine_size_gb:.2f} GB)")
    
    # 选择运行模式
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    
    if mode == "benchmark":
        # 仅测试 Transformer 性能
        print("\n[Mode: Benchmark]")
        run_benchmark(engine_path)
    else:
        # 完整图像生成
        prompt = "A masterpiece photo of a beautiful sunset over rugged mountains, with dramatic, fiery clouds filling the sky. In the foreground, a golden retriever and a fluffy calico cat sit side-by-side on a rocky outcrop, looking out at the view. Cinematic lighting casts long shadows and warm golden light on their fur. 8k resolution, detailed textures."
        print("\n[Mode: Generate Image]")
        image, stats = run_full_pipeline(
            engine_path=engine_path,
            repo_root=REPO_ROOT,
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=3.5,
            seed=42,
            save_path=f"results/tensorrt_output_{PRECISION}.png"
        )
        
        print(f"\nGeneration completed in {stats['generation_time']:.2f}s")