import streamlit as st
from pathlib import Path
import base64

# Configure page
st.set_page_config(
    page_title="Battery Lab - Electrochemical Test Data Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display the cover image directly from file
try:
    # Try direct file path first
    st.image("Cover.jpeg", use_container_width=True)
except Exception as e:
    st.error(f"Unable to display image: {str(e)}")
    st.write("Please check that the Cover.jpeg file is in the same directory as the app.")

# Footer
st.markdown("---")
st.markdown("Battery Lab - Electrochemical Test Data Analyzer | © 2025") 