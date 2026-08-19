import torch
import torchvision.models as models

def main():
    print("Downloading pre-trained MobileNetV2...")
    # We use MobileNetV2 as it's a great lightweight model for edge deployment
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    # Save the PyTorch model weights
    model_path = "mobilenet_v2.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
