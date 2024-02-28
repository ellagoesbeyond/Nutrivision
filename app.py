import streamlit as st
import cv2
import torch
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    # Load the YOLO model
    model_dict = torch.load("best.pt")
    net = model_dict['model']  # Access the model from the dictionary
    return net

# Function to perform object detection
def detect_objects(net, image):
    # Convert image to PyTorch tensor
    frame_tensor = torch.tensor(image)
    # Make predictions
    with torch.no_grad():
        predictions = net.forward(frame_tensor)
    # Process predictions as needed
    # ...
    return predictions

# Streamlit app
def main():
    st.title("Object Detection App")
    
    net = load_model()
    
    if 'snapshot' not in st.session_state:
        st.session_state.snapshot = None

    if 'snapshot_taken' not in st.session_state:
        st.session_state.snapshot_taken = False

    if 'objects_detected' not in st.session_state:
        st.session_state.objects_detected = None

    st.write("Click the button below to take a snapshot:")
    if st.button("Take Snapshot"):
        st.session_state.snapshot_taken = True

    if st.session_state.snapshot_taken:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            st.session_state.snapshot = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
            st.image(st.session_state.snapshot, use_column_width=True)
            if st.button("Detect Objects"):
                objects_detected = detect_objects(net, st.session_state.snapshot)
                st.session_state.objects_detected = objects_detected

    if st.session_state.objects_detected is not None:
        # Process detected objects
        # ...
        st.write("Objects detected!")

if __name__ == "__main__":
    main()

