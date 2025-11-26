import tensorrt as trt
import os
import time
import torch

# 指定 GPU 并初始化 CUDA
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.cuda.set_device(0)
_ = torch.zeros(1, device="cuda")  # 初始化 CUDA context


def build_engine(
    onnx_path, 
    engine_path, 
    max_workspace_size=16,  # 减小默认值
    use_fp16=True,
    verbose=True
):
    """
    将 ONNX 模型转换为 TensorRT Engine
    
    Args:
        onnx_path: ONNX 模型路径
        engine_path: TensorRT Engine 保存路径
        max_workspace_size: 最大工作空间大小（GB）
        use_fp16: 是否使用 FP16 精度
        verbose: 是否打印详细信息
    """
    print("=" * 80)
    print("Building TensorRT Engine")
    print("=" * 80)
    print(f"ONNX model: {onnx_path}")
    print(f"Engine output: {engine_path}")
    print(f"Max workspace: {max_workspace_size} GB")
    print(f"FP16: {use_fp16}")
    print("-" * 80)
    
    # 1. 创建 Logger 和 Builder
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # 2. 创建 Network 定义（显式 Batch）
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    
    # 3. 解析 ONNX 模型
    print("Parsing ONNX model...")
    start_time = time.time()
    
    if not os.path.exists(onnx_path):
        print(f"❌ Error: ONNX model not found at {onnx_path}")
        return False
    
    # 获取 ONNX 文件所在目录（用于加载外部权重）
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    
    # 切换到 ONNX 文件所在目录，确保 TensorRT 能找到外部权重文件
    original_dir = os.getcwd()
    os.chdir(onnx_dir)
    
    try:
        with open(os.path.basename(onnx_path), 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ Failed to parse ONNX model:")
                for error in range(parser.num_errors):
                    print(f"  Error {error}: {parser.get_error(error)}")
                return False
    finally:
        # 切换回原目录
        os.chdir(original_dir)
    
    parse_time = time.time() - start_time
    print(f"✅ ONNX parsed successfully in {parse_time:.2f}s")
    print(f"Network has {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")
    
    # 打印输入信息
    print("\nNetwork Inputs:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: {inp.shape} ({inp.dtype})")
    
    print("\nNetwork Outputs:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: {out.shape} ({out.dtype})")
    print("-" * 80)
    
    # 4. 配置构建参数
    config = builder.create_builder_config()
    
    # 设定最大显存
    workspace_bytes = max_workspace_size * (1 << 30)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    print(f"Workspace memory pool: {max_workspace_size} GB")
    
    # 开启 FP16
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 mode enabled")
    
    # 注意: TensorRT 精度支持说明
    # - FP16: 所有 NVIDIA GPU 支持 (Pascal+)
    # - INT8: 需要校准数据集，用于量化
    #   config.set_flag(trt.BuilderFlag.INT8)
    #   config.int8_calibrator = YourCalibrator()
    # - FP8: 仅 Hopper (H100) 及以上 GPU 支持
    #   需要 TensorRT 9.0+ 和 CUDA 12.0+
    #   config.set_flag(trt.BuilderFlag.FP8)
    elif use_fp16:
        print("⚠️  FP16 requested but not supported on this platform")
    
    # 5. 定义 Optimization Profile（动态 Shape 必须）
    profile = builder.create_optimization_profile()
    
    # 打印实际的输入 shape 以便调试
    print("\nConfiguring optimization profile based on actual input shapes:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = inp.shape
        is_dynamic = any(d == -1 for d in shape)
        print(f"  {inp.name}: {list(shape)} {'(dynamic)' if is_dynamic else '(static)'}")
    
    # 根据实际输入配置 profile
    # hidden_states: [batch, seq_len, 64] - 动态
    profile.set_shape(
        "hidden_states",
        min=(1, 1024, 64),   # 512x512
        opt=(1, 4096, 64),   # 1024x1024
        max=(2, 16384, 64)   # 2x 2048x2048 batch
    )
    
    # encoder_hidden_states: [batch, text_seq_len, 4096] - 动态
    profile.set_shape(
        "encoder_hidden_states",
        min=(1, 256, 4096),
        opt=(1, 512, 4096),
        max=(2, 512, 4096)
    )
    
    # pooled_projections: [batch, 768] - 动态 batch
    profile.set_shape(
        "pooled_projections",
        min=(1, 768),
        opt=(1, 768),
        max=(2, 768)
    )
    
    # timestep: [batch] - 动态 batch
    profile.set_shape(
        "timestep",
        min=(1,),
        opt=(1,),
        max=(2,)
    )
    
    # img_ids: [4096, 3] - 静态（从 ONNX 模型导出时固定）
    # 注意：静态维度不需要设置 profile，或者 min=opt=max
    profile.set_shape(
        "img_ids",
        min=(4096, 3),
        opt=(4096, 3),
        max=(4096, 3)
    )
    
    # txt_ids: [512, 3] - 静态
    profile.set_shape(
        "txt_ids",
        min=(512, 3),
        opt=(512, 3),
        max=(512, 3)
    )
    
    # guidance: [1] - 静态
    profile.set_shape(
        "guidance",
        min=(1,),
        opt=(1,),
        max=(1,)
    )
    
    config.add_optimization_profile(profile)
    print("✅ Optimization profile configured")
    print("-" * 80)
    
    # 6. 构建并序列化 Engine
    print("Building TensorRT Engine (this may take a long time, 10-30 minutes)...")
    print("TensorRT is optimizing the network...")
    
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    
    build_start = time.time()
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ Failed to build engine")
        return False
    
    # 保存引擎
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    build_time = time.time() - build_start
    engine_size_mb = os.path.getsize(engine_path) / (1024 ** 2)
    
    print("=" * 80)
    print("✅ Engine built successfully!")
    print(f"Build time: {build_time / 60:.1f} minutes")
    print(f"Engine size: {engine_size_mb:.2f} MB")
    print(f"Saved to: {engine_path}")
    print("=" * 80)
    
    return True
        
    


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    OUTPUT_DIR = "models"
    MODEL_NAME = "flux_transformer"
    PRECISION = "bf16"  # 需要与 onnx_model.py 导出时使用的精度一致
    # =================================================
    
    # 自动生成文件路径（与 onnx_model.py 保持一致）
    onnx_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{PRECISION}.onnx")
    engine_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{PRECISION}.engine")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Error: ONNX model not found at {onnx_path}")
        print("Please run onnx_model.py first to export the ONNX model.")
        print(f"Expected file: {onnx_path}")
        exit(1)
    
    # 检查权重文件
    weights_path = onnx_path.replace('.onnx', '_weights.bin')
    if os.path.exists(weights_path):
        weights_size_gb = os.path.getsize(weights_path) / (1024 ** 3)
        print(f"Found external weights: {weights_path} ({weights_size_gb:.2f} GB)")
    
    print(f"\n[Building TensorRT Engine ({PRECISION})]")
    success = build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        max_workspace_size=24,  # 24GB
        use_fp16=(PRECISION in ["fp16", "bf16"]),
        verbose=True
    )
    
    if success:
        print("\n✅ TensorRT engine build completed!")
        print(f"   ONNX model: {onnx_path}")
        print(f"   TensorRT engine: {engine_path}")
    else:
        print("\n❌ TensorRT engine build failed!")
        exit(1)