import streamlit as st
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Battery Lab - Electrochemical Test Data Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the absolute path to the image
current_dir = Path(__file__).resolve().parent
image_path = current_dir / "Cover.jpeg"

# Display the cover image with the updated parameter
st.image(str(image_path), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Battery Lab - Electrochemical Test Data Analyzer | © 2025") 