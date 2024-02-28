# Python In-built packages
from pathlib import Path

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="NutriVISION",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Get your nutrition facts with NutriVISION!")

# Sidebar
st.sidebar.header("Navigation")

confidence = 0.4
#float(st.sidebar.slider( "Select Model Confidence", 25, 100, 40)) / 100

model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


helper.play_webcam(confidence, model)
