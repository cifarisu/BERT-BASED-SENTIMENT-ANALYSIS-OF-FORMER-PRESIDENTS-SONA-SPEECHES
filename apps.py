import streamlit as st
from predict_pages import show_predict_pages
from explore_pages import show_explore_pages

# Set default page to "Explore"
page = "Explore"

# Display sidebar to select page
page = st.sidebar.selectbox("Explore or Predict", ("Explore", "Predict"))

# Show appropriate page based on user selection
if page == "Explore":
    show_explore_pages()
else:
    show_predict_pages()
