import streamlit as st
from predict_page import show_predict_page
from sentence_prediction import show_sentence_prediction
from explore_page import show_explore_page
from evaluation_page import show_evaluation_page
from scraper_page import show_scraper_page

# Set title and favicon
st.set_page_config(page_title="SONA Sentiment", page_icon="icon.png")

# Set default page to "Scraper"
page = "Scraper"

# Display sidebar to select page
page = st.sidebar.selectbox("Select an option", ("Scraper", "Speech Prediction", "Single Prediction", "Evaluation", "Presidents"))

# Show appropriate page based on user selection
if page == "Scraper":
    show_scraper_page()
elif page == "Speech Prediction":
    show_predict_page()
elif page == "Single Prediction":
    show_sentence_prediction()
elif page == "Evaluation":
    show_evaluation_page()
else:
    show_explore_page()