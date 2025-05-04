import os

import streamlit as st
import pandas as pd

st.title("Sound File and Annotation Upload")

# Upload sound files
sound_files = st.file_uploader(
    "Upload one or more sound files (e.g., .wav, .mp3)",
    type=["wav", "mp3"],
    accept_multiple_files=True
)

# Upload CSV annotation file
annotation_file = st.file_uploader(
    "Upload annotation file (CSV)",
    type=["csv"]
)

# Show uploaded files
if sound_files:
    st.subheader("Uploaded Sound Files")
    for file in sound_files:
        st.audio(file, format='audio/wav')
        st.write(f"Filename: `{file.name}`")

if annotation_file:
    st.subheader("Annotation File Preview")
    try:
        df = pd.read_csv(annotation_file)
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")

# Validate matching files
if sound_files and annotation_file:
    sound_names = {file.name for file in sound_files}
    try:
        df = pd.read_csv(annotation_file)
        if "Record_Segment" in df.columns:
            annotated_names = set(df["Record_Segment"].unique())
            unmatched = annotated_names - sound_names
            if unmatched:
                st.warning(f"The following files are in the annotation CSV but not uploaded: {unmatched}")
            else:
                st.success("All annotated files are present in the uploaded sounds.")
        else:
            st.error("Annotation CSV must contain a 'filename' column.")
    except Exception as e:
        st.error(f"CSV processing error: {e}")

# Save uploaded audio files
if sound_files:
    for file in sound_files:
        file_path = os.path.join("uploads/audio", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Saved audio: {file.name}")

# Save uploaded annotation file
if annotation_file:
    ann_path = os.path.join("uploads/annotations", annotation_file.name)
    with open(ann_path, "wb") as f:
        f.write(annotation_file.getbuffer())
    st.success(f"Saved annotation file: {annotation_file.name}")