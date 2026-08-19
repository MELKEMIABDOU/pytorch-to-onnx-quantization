from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_onnx_model():
    model_input = "mobilenet_v2.onnx"
    model_output = "mobilenet_v2_quantized.onnx"

    if not os.path.exists(model_input):
        print(f"Error: {model_input} not found. Please run step 2 (export) first.")
        return

    print(f"Quantizing {model_input} dynamically to INT8...")
    
    # Perform dynamic quantization
    # Dynamic quantization converts the weights to INT8, but activations remain in FP32
    # and are quantized on-the-fly during inference. This is easiest to apply and great for CPUs.
    quantize_dynamic(
        model_input=model_input,
        model_output=model_output,
        weight_type=QuantType.QInt8
    )
    
    print(f"Quantized model saved to {model_output}")

if __name__ == "__main__":
    quantize_onnx_model()
