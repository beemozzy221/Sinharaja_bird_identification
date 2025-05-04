import streamlit as st

st.set_page_config(page_title="Sound App", layout="centered")

st.title("Welcome to the Sound Analyzer")

st.markdown("Choose what you'd like to do:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Deposit sound files"):
        st.switch_page("pages/sound_deposit_interface.py")

with col2:
    if st.button("Analyze sound file"):
        st.switch_page("pages/interface.py")