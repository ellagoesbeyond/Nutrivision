import streamlit as st
import cv2
import numpy as np

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

    # Rest of the detection code remains the same...

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
