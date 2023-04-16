import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

# Set default page to "Explore"
page = "Explore"

# Display sidebar to select page
page = st.sidebar.selectbox("Explore or Predict", ("Explore", "Predict"))

# Show appropriate page based on user selection
if page == "Explore":
    show_explore_page()
else:
    show_predict_page()
