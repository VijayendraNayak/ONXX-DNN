import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Export the model
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, "../models/yolov5n.onnx", opset_version=12)