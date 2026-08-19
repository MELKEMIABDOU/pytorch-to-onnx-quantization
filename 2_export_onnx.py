import torch
import torchvision.models as models

def export_to_onnx():
    print("Loading model...")
    # Initialize the same model architecture
    model = models.mobilenet_v2()
    
    # Load the weights
    # Note: weights_only=True is a security best practice when loading untrusted models
    model.load_state_dict(torch.load("mobilenet_v2.pth", weights_only=True))
    model.eval()

    # Create dummy input for the model
    # MobileNetV2 takes 3-channel RGB images of size 224x224
    dummy_input = torch.randn(1, 3, 224, 224)

    onnx_path = "mobilenet_v2.onnx"
    print(f"Exporting model to {onnx_path}...")
    
    # Export the model
    torch.onnx.export(
        model,                               # Model being run
        dummy_input,                         # Model input
        onnx_path,                           # Where to save the model
        export_params=True,                  # Store the trained parameter weights inside the model file
        opset_version=14,                    # The ONNX version to export the model to
        do_constant_folding=True,            # Whether to execute constant folding for optimization
        input_names=['input'],               # the model's input names
        output_names=['output'],             # the model's output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # variable length axes
    )
    print("Export complete!")

if __name__ == "__main__":
    export_to_onnx()
