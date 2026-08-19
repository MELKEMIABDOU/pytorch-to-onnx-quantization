import torch
from model import SimpleGRU
import onnxruntime as ort
import numpy as np
import time
import os

def get_model_size_mb(file_path):
    if not os.path.exists(file_path): 
        return 0
    return os.path.getsize(file_path) / (1024 * 1024)

def benchmark_pytorch(model, input_tensor, iterations=100):
    print("Benchmarking PyTorch...")
    
    # Warmup runs
    for _ in range(10): 
        _ = model(input_tensor)
        
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_tensor)
    return (time.time() - start) / iterations

def benchmark_onnx(model_path, input_array, iterations=100):
    if not os.path.exists(model_path): 
        return 0
        
    print(f"Benchmarking ONNX ({model_path})...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Warmup runs
    for _ in range(10): 
        _ = session.run(None, {input_name: input_array})
        
    start = time.time()
    for _ in range(iterations):
        _ = session.run(None, {input_name: input_array})
    return (time.time() - start) / iterations

def main():
    pth_path = "simple_gru.pth"
    onnx_path = "simple_gru.onnx"
    onnx_fp16_path = "simple_gru_fp16.onnx"
    onnx_int8_path = "simple_gru_int8.onnx"
    
    if not os.path.exists(onnx_int8_path):
        print("Run scripts 1, 2, and 3 first.")
        return
    
    print("\n==============================")
    print("      Model Size Comparison   ")
    print("==============================")
    print(f"PyTorch (.pth):      {get_model_size_mb(pth_path):.2f} MB")
    print(f"ONNX (FP32):         {get_model_size_mb(onnx_path):.2f} MB")
    print(f"ONNX (FP16):         {get_model_size_mb(onnx_fp16_path):.2f} MB")
    print(f"ONNX (INT8):         {get_model_size_mb(onnx_int8_path):.2f} MB")
    print("\n")
    
    # Generate dummy input tensor
    tensor_input = torch.randn(1, 128, 24)
    numpy_input = tensor_input.numpy()
    
    # Load PyTorch model
    model = SimpleGRU()
    model.load_state_dict(torch.load(pth_path, weights_only=True))
    model.eval()
    
    print("==============================")
    print("  Inference Speed Comparison  ")
    print("==============================")
    pytorch_time = benchmark_pytorch(model, tensor_input) * 1000
    onnx_time = benchmark_onnx(onnx_path, numpy_input) * 1000
    onnx_fp16_time = benchmark_onnx(onnx_fp16_path, numpy_input) * 1000
    onnx_int8_time = benchmark_onnx(onnx_int8_path, numpy_input) * 1000
    
    print("\n--- Results (ms/iter) ---")
    print(f"PyTorch:             {pytorch_time:.2f} ms")
    print(f"ONNX (FP32):         {onnx_time:.2f} ms")
    print(f"ONNX (FP16):         {onnx_fp16_time:.2f} ms")
    print(f"ONNX (INT8 Quant):   {onnx_int8_time:.2f} ms")

if __name__ == "__main__":
    main()
