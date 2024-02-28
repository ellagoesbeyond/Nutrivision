from ultralytics import YOLO
import time
import streamlit as st
import cv2


import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


"""def display_tracker_options():
    display_tracker = st.toggle( "Let's Start!")
    is_display_tracker = True if display_tracker == "Let's Start!" else False
    tracker = "bytetrack.yaml"
    return is_display_tracker,tracker"""


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))
    trackeri = "bytetrack.yaml"

    #if is_display_tracking:
    res = model.track(image, conf=conf, persist=True, tracker=trackeri)
    
    
    """else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
        print(res)"""

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    print(res_plotted)
    st_frame.image(res_plotted,
                   caption='We see you got ..',
                   channels="BGR",
                   use_column_width=True
                   )




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
    #is_display_tracker, tracker = display_tracker_options()
    col1,col2=st.columns(2)
    with col1:
        go = st.button("GO")
    with  col2:
        stop = st.button("STOP")

    if go:
        try:
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image)
                     
            if stop:
                vid_cap.release()
        except:
            st.error("Unable to open webcam. Please check the webcam connection.")               
              

      
       
