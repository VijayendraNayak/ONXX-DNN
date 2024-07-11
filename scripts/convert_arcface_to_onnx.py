import torch
import torchvision.models as models

try:
    # Load the ResNet50 model
    model = models.resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    print("Model loaded successfully")

    # Export the model
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "../models/resnet50.onnx", opset_version=12)

    print("ONNX export completed successfully")

except Exception as e:
    print(f"Error:{e}")
