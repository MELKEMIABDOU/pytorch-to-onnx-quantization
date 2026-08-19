import torch
from model import SimpleGRU

def export_to_onnx():
    print("Loading GRU model...")
    model = SimpleGRU()
    
    # Load weights
    model.load_state_dict(torch.load("simple_gru.pth", weights_only=True))
    model.eval()

    # Generate dummy input matching telemetry shape
    # Shape: (batch_size=1, sequence_length=128, input_features=24)
    dummy_input = torch.randn(1, 128, 24)

    onnx_path = "simple_gru.onnx"
    print(f"Exporting model to {onnx_path}...")
    
    # Export model to ONNX format
    torch.onnx.export(
        model,                               # Target model
        dummy_input,                         # Dummy input tensor
        onnx_path,                           # Output file path
        export_params=True,                  # Embed weights in the ONNX file
        opset_version=14,                    # Use opset 14 for optimal RNN support
        do_constant_folding=True,            # Optimize graph with constant folding
        input_names=['input_telemetry'],     # Define input node name
        output_names=['output'],             # Define output node name
        dynamic_axes={'input_telemetry': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete!")

if __name__ == "__main__":
    export_to_onnx()
