import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics immport YOLO

# Load YOLO model
net = cv2.dnn.readNet("best.pt")


CLASSES = [
    "Apple", "Banana", "Beetroot", "Bitter_Gourd", "Bottle_Gourd", "Cabbage",
    "Capsicum", "Carrot", "Cauliflower", "Cherry", "Chilli", "Coconut", 
    "Cucumber", "EggPlant", "Ginger", "Grape", "Green_Orange", "Kiwi", 
    "Maize", "Mango", "Melon", "Okra", "Onion", "Orange", "Peach", "Pear", 
    "Peas", "Pineapple", "Pomegranate", "Potato", "Radish", "Strawberry", 
    "Tomato", "Turnip", "Watermelon", "almond", "walnut"
]

# Function to perform object detection
def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    height, width, _ = image.shape

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = CLASSES[class_ids[i]]
            confidence = confidences[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

st.title("Object Detection with YOLO")

# Function to capture images from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to open webcam.")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame

if st.button("Capture Image"):
    frame = capture_image()
    if frame is not None:
        st.image(frame, caption="Captured Image", use_column_width=True)
        detected_frame = detect_objects(frame)
        st.image(detected_frame, caption="Detected Objects", use_column_width=True)
