import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

# Set title and favicon
st.set_page_config(page_title="SONA Sentiment", page_icon="icon.png")

# Set default page to "Predict
page = "Predict"

# Display sidebar to select page
page = st.sidebar.selectbox("Predict or Explore", ("Predict", "Explore"))

# Show appropriate page based on user selection
if page == "Predict":
    show_predict_page()
else:
    show_explore_page()