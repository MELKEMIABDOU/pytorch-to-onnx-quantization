import torch
import torchvision.models as models
import onnxruntime as ort
import numpy as np
import time
import os

def get_model_size_mb(file_path):
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def benchmark_pytorch(model, input_tensor, iterations=100):
    print("Benchmarking PyTorch...")
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
        
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_tensor)
    end = time.time()
    
    return (end - start) / iterations

def benchmark_onnx(model_path, input_array, iterations=100):
    print(f"Benchmarking ONNX ({model_path})...")
    # Load model using ONNX Runtime
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: input_array})
        
    start = time.time()
    for _ in range(iterations):
        _ = session.run(None, {input_name: input_array})
    end = time.time()
    
    return (end - start) / iterations

def main():
    pth_path = "mobilenet_v2.pth"
    onnx_path = "mobilenet_v2.onnx"
    onnx_quant_path = "mobilenet_v2_quantized.onnx"
    
    if not os.path.exists(onnx_quant_path):
        print("Please run scripts 1, 2, and 3 first to generate the models.")
        return
    
    print("==============================")
    print("      Model Size Comparison   ")
    print("==============================")
    print(f"PyTorch (.pth):      {get_model_size_mb(pth_path):.2f} MB")
    print(f"ONNX (FP32):         {get_model_size_mb(onnx_path):.2f} MB")
    print(f"ONNX (INT8 Quant):   {get_model_size_mb(onnx_quant_path):.2f} MB")
    print("\n")
    
    # Dummy inputs for inference testing
    tensor_input = torch.randn(1, 3, 224, 224)
    numpy_input = tensor_input.numpy()
    
    # Load PyTorch model for benchmarking
    model = models.mobilenet_v2()
    model.load_state_dict(torch.load(pth_path, weights_only=True))
    model.eval()
    
    # Run benchmarks
    print("==============================")
    print("  Inference Speed Comparison  ")
    print("==============================")
    pytorch_time = benchmark_pytorch(model, tensor_input) * 1000  # Convert to ms
    onnx_time = benchmark_onnx(onnx_path, numpy_input) * 1000
    onnx_quant_time = benchmark_onnx(onnx_quant_path, numpy_input) * 1000
    
    print("\n--- Results (ms/iter) ---")
    print(f"PyTorch:             {pytorch_time:.2f} ms")
    print(f"ONNX (FP32):         {onnx_time:.2f} ms")
    print(f"ONNX (INT8 Quant):   {onnx_quant_time:.2f} ms")

if __name__ == "__main__":
    main()
