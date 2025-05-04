import streamlit as st
import os

# Admin login
st.sidebar.header("Admin Login")
entered_password = st.sidebar.text_input("Enter admin password", type="password")

if entered_password == st.secrets["admin_password"]:
    st.success("Admin access granted ✅")

    # List downloadable audio files
    audio_folder = "uploads/audio"
    if os.path.exists(audio_folder):
        for filename in os.listdir(audio_folder):
            file_path = os.path.join(audio_folder, filename)
            with open(file_path, "rb") as f:
                st.download_button(label=f"Download {filename}", data=f, file_name=filename)
    else:
        st.warning("No audio files found.")

    # List downloadable annotation files
    annot_folder = "uploads/annotations"
    if os.path.exists(annot_folder):
        for filename in os.listdir(annot_folder):
            file_path = os.path.join(annot_folder, filename)
            with open(file_path, "rb") as f:
                st.download_button(label=f"Downlosad {filename}", data=f, file_name=filename)
    else:
        st.warning("No annotation files found.")

elif entered_password:
    st.error("Incorrect password ❌")