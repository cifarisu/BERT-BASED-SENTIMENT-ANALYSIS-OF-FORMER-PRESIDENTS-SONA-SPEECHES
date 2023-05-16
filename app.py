import streamlit as st
from predict_page import show_predict_page
from sentence_prediction import show_sentence_prediction
from explore_page import show_explore_page

# Set title and favicon
st.set_page_config(page_title="SONA Sentiment", page_icon="icon.png")

# Set default page to "Speech Prediction"
page = "Speech Prediction"

# Display sidebar to select page
page = st.sidebar.selectbox("Select an option", ("Speech Prediction", "Single Prediction", "Presidents"))

# Show appropriate page based on user selection
if page == "Speech Prediction":
    show_predict_page()
elif page == "Single Prediction":
    show_sentence_prediction()
else:
    show_explore_page()