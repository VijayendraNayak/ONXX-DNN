import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import onnxruntime as ort
import cv2
import os

# Absolute paths
face_detector_model_path = os.path.abspath('../models/yolov5n.onnx')
face_recognition_model_path = os.path.abspath('../models/resnet50.onnx')

print(f"Using face detector model at: {face_detector_model_path}")
print(f"Using face recognition model at: {face_recognition_model_path}")

if not os.path.exists(face_detector_model_path):
    raise FileNotFoundError(f"Face detector model file not found: {face_detector_model_path}")
if not os.path.exists(face_recognition_model_path):
    raise FileNotFoundError(f"Face recognition model file not found: {face_recognition_model_path}")

face_detector = ort.InferenceSession(face_detector_model_path)
face_recognizer = ort.InferenceSession(face_recognition_model_path)

# Define a function to perform face detection
def detect_faces(image, model):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    # Preprocess image as per YOLOv5 requirements
    # This is just an example; adjust preprocessing as per your model's requirements
    image = cv2.resize(image, (640, 640))
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = image[np.newaxis, :, :, :] / 255.0  # Normalize
    return model.run([output_name], {input_name: image.astype(np.float32)})[0]

# Define a function to perform face recognition
def recognize_faces(face_images, model):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    face_embeddings = []
    for face in face_images:
        print(f"Processing face with shape: {face.shape}")  # Debugging statement
        # Ensure the image has 3 channels (convert if necessary)
        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 1:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        # Resize and normalize
        face = cv2.resize(face, (224, 224))
        print(f"Resized face shape: {face.shape}")  # Debugging statement
        face = face.transpose(2, 0, 1)  # HWC to CHW
        print(f"Transposed face shape: {face.shape}")  # Debugging statement
        face = face[np.newaxis, :, :, :] / 255.0  # Normalize
        print(f"Normalized face shape: {face.shape}")  # Debugging statement

        # Ensure input dimensions match model requirements
        assert face.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {face.shape}"

        embedding = model.run([output_name], {input_name: face.astype(np.float32)})[0]
        face_embeddings.append(embedding)
    return face_embeddings

# Placeholder function to match recognized faces to labels
def match_faces_to_labels(face_embeddings):
    # Implement your logic here to match face embeddings to labels
    # This is a placeholder; replace with your actual matching logic
    predicted_labels = []
    for embedding in face_embeddings:
        # Example: If embedding meets a threshold, classify as a specific label
        if np.random.rand() > 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(2)
    return predicted_labels

# Load your images and corresponding ground truth labels
images = [cv2.imread(f'../data/{filename}') for filename in ['ronaldo1.png', 'ronaldo2.png', 'virat1.png', 'virat2.png']]
ground_truth_labels = np.array([1, 1, 2, 2])  # Example ground truth labels, replace with actual

# Make predictions
predicted_labels = []
for image in images:
    detected_faces = detect_faces(image, face_detector)
    recognized_faces = recognize_faces(detected_faces, face_recognizer)
    # Assuming you have a way to match recognized faces to labels
    # Use the placeholder function to match recognized faces to labels
    predicted_labels.extend(match_faces_to_labels(recognized_faces))

predicted_labels = np.array(predicted_labels)

# Debugging output to inspect labels
print(f"Ground Truth Labels: {ground_truth_labels}")
print(f"Predicted Labels: {predicted_labels}")

# Print unique labels to identify any issues
print(f"Unique Ground Truth Labels: {np.unique(ground_truth_labels)}")
print(f"Unique Predicted Labels: {np.unique(predicted_labels)}")

# Calculate metrics
precision = precision_score(ground_truth_labels, predicted_labels, average='weighted')
recall = recall_score(ground_truth_labels, predicted_labels, average='weighted')
f1 = f1_score(ground_truth_labels, predicted_labels, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
