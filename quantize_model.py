from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
from onnxconverter_common import float16
import os

def quantize_to_int8(model_input, model_output):
    print(f"Quantizing {model_input} dynamically to INT8...")
    
    # Apply dynamic INT8 quantization for CPU inference optimization
    quantize_dynamic(
        model_input=model_input,
        model_output=model_output,
        weight_type=QuantType.QInt8
    )
    print(f"INT8 model saved to {model_output}")

def quantize_to_fp16(model_input, model_output):
    print(f"Quantizing {model_input} statically to FP16...")
    
    # Convert model to FP16 to reduce size for GPU/NPU deployment
    onnx_model = onnx.load(model_input)
    fp16_model = float16.convert_float_to_float16(onnx_model)
    onnx.save(fp16_model, model_output)
    print(f"FP16 model saved to {model_output}")

def main():
    model_input = "simple_gru.onnx"
    
    if not os.path.exists(model_input):
        print(f"Error: {model_input} not found. Run export script first.")
        return

    quantize_to_int8(model_input, "simple_gru_int8.onnx")
    print("-" * 30)
    quantize_to_fp16(model_input, "simple_gru_fp16.onnx")

if __name__ == "__main__":
    main()
