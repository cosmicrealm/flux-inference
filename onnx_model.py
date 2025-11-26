import torch
import os
import time
import tempfile
import shutil
import onnxruntime as ort
import numpy as np
from diffusers import FluxPipeline
from torch.onnx import register_custom_op_symbolic
import onnx
from onnx.external_data_helper import convert_model_to_external_data

# 精度配置
# 注意:
# - fp32/fp16/bf16: 标准精度，ONNX Runtime 完全支持
# - fp8/fp4: 需要特殊的量化工具链，ONNX Runtime 原生不支持
#   - FP8: 需要 NVIDIA Transformer Engine 或 TensorRT 9+ (Hopper GPU)
#   - FP4/INT4: 需要使用 GPTQ/AWQ 等量化方法，然后用专门的推理引擎
PRECISION_MAP = {
    "fp32": {"torch_dtype": torch.float32, "np_dtype": np.float32},
    "fp16": {"torch_dtype": torch.float16, "np_dtype": np.float16},
    "bf16": {"torch_dtype": torch.bfloat16, "np_dtype": np.float16},
    # FP8/FP4 需要额外的量化步骤，这里仅作标记
    # "fp8": 需要 Transformer Engine 量化后导出
    # "fp4": 需要 GPTQ/AWQ 量化后导出
}


def get_output_path(output_dir: str, model_name: str, precision: str) -> str:
    """根据精度生成输出路径"""
    return os.path.join(output_dir, f"{model_name}_{precision}.onnx")
# 定义 rms_norm 的符号化函数
def rms_norm_symbolic(g, input, normalized_shape, weight, eps):
    """
    将 rms_norm 分解为 ONNX 支持的基础操作
    RMSNorm(x) = x / RMS(x) * weight
    其中 RMS(x) = sqrt(mean(x^2) + eps)
    """
    # 计算 x^2
    square = g.op("Mul", input, input)
    
    # 计算 mean(x^2)，在最后一个维度上
    mean = g.op("ReduceMean", square, axes_i=[-1], keepdims_i=1)
    
    # 加上 eps（如果 eps 是符号值，直接使用；如果是常量，需要转换）
    mean_eps = g.op("Add", mean, eps)
    
    # 计算 sqrt
    rms = g.op("Sqrt", mean_eps)
    
    # x / rms
    normalized = g.op("Div", input, rms)
    
    # * weight
    if weight is not None:
        result = g.op("Mul", normalized, weight)
    else:
        result = normalized
    
    return result

# 注册自定义符号化函数
register_custom_op_symbolic('aten::rms_norm', rms_norm_symbolic, opset_version=17)


def export_transformer_to_onnx(model_path: str, output_dir: str, model_name: str, precision: str):
    """导出 FLUX Transformer 到 ONNX 格式"""
    if precision not in PRECISION_MAP:
        raise ValueError(f"Unsupported precision: {precision}. Supported: {list(PRECISION_MAP.keys())}")
    
    dtype = PRECISION_MAP[precision]["torch_dtype"]
    output_path = get_output_path(output_dir, model_name, precision)
    
    print("=" * 80)
    print(f"Exporting FLUX Transformer to ONNX ({precision})")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print("-" * 80)
    
    # 1. 加载 Transformer 组件
    print("Loading FLUX pipeline...")
    start_time = time.time()
    
    load_dtype = torch.bfloat16 if precision == "bf16" else dtype
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=load_dtype)
    transformer = pipe.transformer
    transformer.eval()
    
    # bf16 转换为 fp16 导出（ONNX Runtime 对 bf16 支持有限）
    if precision == "bf16":
        transformer = transformer.to(torch.float16)
    
    transformer = transformer.cpu()
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    print("-" * 80)
    
    # 2. 构造 Dummy Input
    print("Preparing dummy inputs...")
    export_dtype = torch.float16 if precision in ["fp16", "bf16"] else torch.float32
    
    # 1024x1024 图片的 latent: 128x128，packed 后约 4096 tokens
    hidden_states = torch.randn(1, 4096, 64, dtype=export_dtype)
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=export_dtype)
    pooled_projections = torch.randn(1, 768, dtype=export_dtype)
    timestep = torch.tensor([1.0], dtype=export_dtype)
    img_ids = torch.randn(4096, 3, dtype=export_dtype)
    txt_ids = torch.randn(512, 3, dtype=export_dtype)
    guidance = torch.tensor([3.5], dtype=export_dtype)

    dummy_inputs = (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance
    )
    
    input_names = [
        "hidden_states", 
        "encoder_hidden_states", 
        "pooled_projections", 
        "timestep", 
        "img_ids", 
        "txt_ids", 
        "guidance"
    ]
    output_names = ["sample"]

    # 3. 定义动态轴
    dynamic_axes = {
        "hidden_states": {0: "batch", 1: "seq_len"},
        "encoder_hidden_states": {0: "batch", 1: "text_seq_len"},
        "pooled_projections": {0: "batch"},
        "timestep": {0: "batch"},
        "sample": {0: "batch", 1: "seq_len"}
    }

    # 4. 导出 ONNX（先导出到临时目录，避免中间文件污染）
    print("Exporting to ONNX (this may take several minutes)...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建临时目录保存中间文件
    temp_dir = tempfile.mkdtemp(prefix="onnx_export_")
    temp_onnx_path = os.path.join(temp_dir, os.path.basename(output_path))
    
    export_start = time.time()
    
    try:
        torch.onnx.export(
            transformer,
            dummy_inputs,
            temp_onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
            keep_initializers_as_inputs=False,
        )
        export_time = time.time() - export_start
        
        # 对于大模型，需要使用外部数据格式
        onnx_filename = os.path.basename(output_path)
        external_data_path = onnx_filename.replace('.onnx', '_weights.bin')
        
        print("Converting to external data format (required for large models)...")
        
        # 加载模型（包括已有的外部数据）
        model = onnx.load(temp_onnx_path, load_external_data=True)
        
        # 转换为单个外部数据文件
        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=external_data_path,
            size_threshold=1024,
            convert_attribute=False
        )
        
        # 保存最终模型到输出目录
        onnx.save_model(model, output_path)
        
        print(f"Model saved with external data: {external_data_path}")
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp directory: {temp_dir}")
    
    # 获取文件大小
    onnx_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    weights_path = os.path.join(output_dir, external_data_path)
    weights_size_mb = os.path.getsize(weights_path) / (1024 ** 2) if os.path.exists(weights_path) else 0
    file_size_mb = onnx_size_mb + weights_size_mb
    
    print(f"✅ Export successful!")
    print(f"Export time: {export_time:.2f}s")
    print(f"Precision: {precision}")
    print(f"ONNX graph size: {onnx_size_mb:.2f} MB")
    print(f"Weights file size: {weights_size_mb:.2f} MB")
    print(f"Total size: {file_size_mb:.2f} MB")
    print(f"Saved to: {output_path}")
    print("=" * 80)
    
    return output_path


def test_onnx_inference(output_dir: str, model_name: str, precision: str):
    """测试 ONNX 模型推理"""
    onnx_path = get_output_path(output_dir, model_name, precision)
    np_dtype = np.float16 if precision in ["fp16", "bf16"] else np.float32
    
    print("=" * 80)
    print(f"Testing ONNX Inference ({precision})")
    print("=" * 80)
    print(f"Model path: {onnx_path}")
    
    # 检查外部数据文件
    onnx_dir = os.path.dirname(onnx_path) or "."
    external_data_path = os.path.join(onnx_dir, os.path.basename(onnx_path).replace('.onnx', '_weights.bin'))
    if os.path.exists(external_data_path):
        weights_size_gb = os.path.getsize(external_data_path) / (1024 ** 3)
        print(f"External weights: {external_data_path} ({weights_size_gb:.2f} GB)")
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"Execution providers: {providers}")
    print("-" * 80)
    
    # 创建推理会话
    print("Creating ONNX Runtime session...")
    start_time = time.time()
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    print(f"Session created in {time.time() - start_time:.2f}s")
    print(f"Available providers: {session.get_providers()}")
    print("-" * 80)
    
    # 准备测试输入
    print(f"Preparing test inputs (dtype: {np_dtype})...")
    inputs = {
        "hidden_states": np.random.randn(1, 4096, 64).astype(np_dtype),
        "encoder_hidden_states": np.random.randn(1, 512, 4096).astype(np_dtype),
        "pooled_projections": np.random.randn(1, 768).astype(np_dtype),
        "timestep": np.array([1.0], dtype=np_dtype),
        "img_ids": np.random.randn(4096, 3).astype(np_dtype),
        "txt_ids": np.random.randn(512, 3).astype(np_dtype),
        "guidance": np.array([3.5], dtype=np_dtype)
    }
    
    # 预热
    print("Warming up...")
    _ = session.run(None, inputs)
    
    # 正式推理
    print("Running inference...")
    start_time = time.time()
    outputs = session.run(None, inputs)
    inference_time = time.time() - start_time
    
    print(f"✅ Inference successful!")
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output dtype: {outputs[0].dtype}")
    print(f"Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
    print("=" * 80)
    
    return outputs


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    REPO_ROOT = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-dev"
    OUTPUT_DIR = "models"
    MODEL_NAME = "flux_transformer"
    PRECISION = "bf16"  # 可选: "fp32", "fp16", "bf16" "int8"
    # =================================================
    
    print(f"Using ONNX Runtime version: {ort.__version__}")
    print(f"\n🚀 FLUX ONNX Export & Test (precision: {PRECISION})\n")
    
    # Step 1: 导出
    onnx_path = export_transformer_to_onnx(REPO_ROOT, OUTPUT_DIR, MODEL_NAME, PRECISION)
    
    # Step 2: 测试
    if onnx_path and os.path.exists(onnx_path):
        test_onnx_inference(OUTPUT_DIR, MODEL_NAME, PRECISION)
    
    print("\n✅ Done!")