import streamlit as st

# Configure page
st.set_page_config(
    page_title="Battery Lab - Electrochemical Test Data Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display the image using HTML with GitHub raw content URL
st.markdown("""
<div style="display: flex; justify-content: center;">
    <img src="https://raw.githubusercontent.com/jrjfonseca/Simple_Plotter/main/Cover.jpeg" style="width: 100%;">
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Battery Lab - Electrochemical Test Data Analyzer | © 2025") 