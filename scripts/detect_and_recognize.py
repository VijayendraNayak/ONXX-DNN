import os
import cv2
import numpy as np
import onnxruntime as ort

# Print the current working directory
print("Current working directory:", os.getcwd())

# Load the YOLOv5 model
yolov5_model_path = "../models/yolov5n.onnx"
yolov5_session = ort.InferenceSession(yolov5_model_path)

# Load the ArcFace model
arcface_model_path = "../models/resnet50.onnx"
arcface_session = ort.InferenceSession(arcface_model_path)

# Function to run YOLOv5 model and detect faces
def detect_faces(image):
    input_size = (640, 640)
    image_resized = cv2.resize(image, input_size)
    image_transposed = image_resized.transpose(2, 0, 1)
    image_normalized = image_transposed / 255.0
    input_tensor = np.expand_dims(image_normalized, axis=0).astype(np.float32)
    
    input_name = yolov5_session.get_inputs()[0].name
    outputs = yolov5_session.run(None, {input_name: input_tensor})
    
    return outputs[0]

# Function to run ArcFace model and extract face embeddings
def get_face_embedding(face_image):
    face_resized = cv2.resize(face_image, (112, 112))
    face_normalized = face_resized / 255.0
    face_transposed = face_normalized.transpose(2, 0, 1)
    face_input_tensor = np.expand_dims(face_transposed, axis=0).astype(np.float32)
    
    input_name = arcface_session.get_inputs()[0].name
    embedding = arcface_session.run(None, {input_name: face_input_tensor})
    
    return embedding[0]

# Load an image
image_path = "../data/ronaldo1.png"

if os.path.exists(image_path):
    print(f"Image found at {image_path}")
    image = cv2.imread(image_path)
else:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Detect faces
detected_faces = detect_faces(image)

# Print the structure of detected faces to debug
print("Detected faces structure:", detected_faces)

# Post-process YOLOv5 output
def process_yolov5_output(output, img_shape, conf_threshold=0.5):
    boxes = []
    for detection in output[0]:
        x_center, y_center, width, height, conf, *class_probs = detection
        if conf > conf_threshold:
            x1 = int((x_center - width / 2) * img_shape[1])
            y1 = int((y_center - height / 2) * img_shape[0])
            x2 = int((x_center + width / 2) * img_shape[1])
            y2 = int((y_center + height / 2) * img_shape[0])
            cls = np.argmax(class_probs)
            boxes.append((x1, y1, x2, y2, conf, cls))
    return boxes

# Process the detected faces
img_shape = image.shape
processed_faces = process_yolov5_output(detected_faces, img_shape)

# Iterate through detected faces and get embeddings
face_embeddings = []
for x1, y1, x2, y2, conf, cls in processed_faces:
    face_image = image[y1:y2, x1:x2]
    embedding = get_face_embedding(face_image)
    face_embeddings.append((embedding, (x1, y1, x2, y2)))

# Draw rectangles around detected faces
for embedding, (x1, y1, x2, y2) in face_embeddings:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the image with detected faces
output_path = "../data/output_image.png"
cv2.imwrite(output_path, image)
print(f"Output image saved to {output_path}")
