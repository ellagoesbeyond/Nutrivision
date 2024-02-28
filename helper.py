from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import pandas as pd 

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

    
    res = model.track(image, conf=conf, persist=True, tracker=trackeri)
    # Iterate over each detected object
    for det in res.xyxy[0]:
        # Extract class and label information
        class_id = int(det[5])
        label = model.names[class_id]

        # Print class and label information
        print(f"Detected {label} with confidence {det[4]}")

    
    """else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
        print(res)"""

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    
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
    results = {}
    source_webcam = settings.WEBCAM_PATH
    
  
    go = st.sidebar.button("GO")
    
    stop = st.sidebar.button("STOP")

    if go:
        try:
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            temp_df = pd.DataFrame(columns=['class_id', 'label'])
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image)
                     
            if stop:
                cv2.imwrite("pics/Screenshot.png", image)
                st.success("Screenshot saved successfully.")
                vid_cap.release()
                # Read the saved image
                img = cv2.imread("pics/Screenshot.png")
                st.frame.image(img, channels="BGR", use_column_width=True)
                res = model.predict(img, conf=conf)[0]
                st.write(res)
                """if isinstance(res, list):
                    # If multiple detection results are returned
                    for r in res:
                         # Iterate over each detected object
                        for det in r:
                            # Extract class and label information
                            class_id = int(det[5])
                            label = model.names[class_id]

                            # Print class and label information
                            st.write(f"Detected {label} with confidence {det[4]}")

                                # Append detected object information to temp_df
                            temp_df = temp_df.append({'class_id': class_id, 'label': label}, ignore_index=True)
                else:
                    # If single detection result is returned
                    for det in res:
                    # Extract class and label information
                        class_id = int(det[5])
                        label = model.names[class_id]

                        # Print class and label information
                        st.write(f"Detected {label} with confidence {det[4]}")

                        # Append detected object information to temp_df
                        temp_df = temp_df.append({'class_id': class_id, 'label': label}, ignore_index=True)

                    # Construct the 'results' dictionary
                results = {
                            'ingredients': list(temp_df['label']),  # List of detected labels # Placeholder for type
                            'amount_of': temp_df['label'].value_counts().to_dict().items()  # Count the amount of each label
                    }

                        # Print the results
                st.write("Results:", results)"""

        except Exception as e:
            st.error(f"Error: {e}")
            results = {}               
    return results        

      
       
