import streamlit as st
import cv2
import settings  # Assuming you have a settings module with constants
  # Assuming you have a function for displaying tracker options
from yolo_model import YOLOv8  # Assuming you have a YOLOv8 model class
from utils import display_detected_frames  # Assuming you have a function for displaying detected frames
import settings
import helper

def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = helper.display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

# Assuming the necessary functions and classes are imported correctly

def main():
    st.title("Webcam Object Detection")
    
    # Load YOLOv8 model
    model = YOLOv8()  # Instantiate your YOLOv8 model
    
    # Set confidence threshold
    conf = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    play_webcam(conf, model)

if __name__ == "__main__":
    main()
