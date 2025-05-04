import streamlit as st
from streamlit_lottie import st_lottie
import requests

#------------ChatGPT generated interface------------

st.set_page_config(
    page_title="Sinharaja Bird Sound ID",
    page_icon="🕊️",
    layout="centered"
)


# Function to load Lottie animation from a URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Load animations
forest_anim = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_w51pcehl.json")  # Forest animation
bird_anim = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_gjmecwii.json")  # Bird flying animation

# --- Main Interface ---
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🕊️ Sinharaja Bird Sound Identification</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: center; color: #4F7942;'>A machine learning project to explore the voices of Sinharaja rainforest birds</h4>",
    unsafe_allow_html=True)

st_lottie(forest_anim, speed=1, height=250, key="forest")

st.markdown(
    """
    Welcome to the **Sinharaja Bird Sound Identification** system — an AI-powered tool designed to identify bird species based on their songs and calls, recorded deep within the lush Sinharaja rainforest of Sri Lanka.

    With the help of bioacoustic analysis and deep learning, this platform supports both citizen scientists and researchers in managing and understanding bird audio data.
    """
)

st.markdown("---")

st.markdown("### 🌿 What can you do here?")

col1, col2 = st.columns(2)

with col1:
    st_lottie(bird_anim, height=150, key="bird")

with col2:
    st.markdown("""
    **1. Deposit Your Own Annotation Files**  
    Upload your bird audio annotations (e.g., .txt or .csv) to contribute to the growing dataset.

    **2. Process Sound Files**  
    Submit bird audio recordings for automatic species prediction using our trained ML model.
    """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Explore the rainforest, one song at a time 🌱</p>",
            unsafe_allow_html=True)
