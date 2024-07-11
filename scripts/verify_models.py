import os

# Check contents of the models directory
models_dir = '../models'
print(f"Contents of {models_dir}: {os.listdir(models_dir)}")

# Check for specific files
face_detector_model_path = os.path.join(models_dir, 'yolov5n.onnx')
face_recognition_model_path = os.path.join(models_dir, 'resnet50.onnx')

if not os.path.exists(face_detector_model_path):
    raise FileNotFoundError(f"Face detector model file not found: {face_detector_model_path}")
if not os.path.exists(face_recognition_model_path):
    raise FileNotFoundError(f"Face recognition model file not found: {face_recognition_model_path}")

print("Both model files found successfully.")
