import librosa
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import model
import waveletdecomp
from os.path import join as pjoin

# --------------CHATGPT GENERATED CODE (with some minor mods)----------------

def predict_format(audio_array):
    predict_data = []

    #For each second
    for i in range(0, audio_array.shape[0]):
        # Decompose the wav file
        packet = waveletdecomp.wavpacketdecomp(audio_array[i])

        #Collect the coefficients
        _, nodes, _ = waveletdecomp.collect_coefficients(packet)

        #Append to list
        predict_data.append(nodes)

    predict_data = np.array(predict_data)
    predict_data = predict_data.reshape(1, *predict_data.shape, 1)

    return predict_data

# Parameters (replace with actual values or compute dynamically)
dropout_rate = 0.25
hidden_units = [512, 512]
lstm_hidden_units = [128,128]
filter_size = [32, 32, 32]

# ========== CONFIG ==========
cpr = os.path.dirname(__file__)
mpr = os.path.dirname(cpr)
WEIGHTS_DIR = "weights"
SAMPLE_RATE = 44100

# ========== STREAMLIT APP ==========
st.title("🐦 Bird Sound Identifier")

# Bird selection
available_birds = [os.path.splitext(f)[0] for f in os.listdir(pjoin(mpr, WEIGHTS_DIR)) if f.endswith(".weights.h5")]
bird_choice = st.multiselect("Select bird(s) to identify", available_birds, default=available_birds[:1])

uploaded_file = st.file_uploader("Upload a WAV file", type=['wav', 'mp3'])

if uploaded_file is not None and bird_choice:
    st.audio(uploaded_file, format='audio/wav/mp3')

    # Spectrogram Generation
    if st.checkbox("Generate spectrogram?"):
        # Load audio
        spec_data, spec_sr = waveletdecomp.read_audio_files(uploaded_file)
        print(f"RAW FILE SHAPE: {spec_data.shape}")

        # Generate spectrogram
        st.write("Spectrogram:")
        fig, ax = plt.subplots()
        S = librosa.stft(spec_data)
        S_db = librosa.amplitude_to_db(abs(S))
        img = librosa.display.specshow(S_db, sr=spec_sr, x_axis='time', y_axis='hz', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        # Plot
        st.pyplot(fig)
        uploaded_file.seek(0)

    # Load audio
    audio = waveletdecomp.waveletdecomp_interface(uploaded_file, SAMPLE_RATE)
    print(f"Loaded audio information: {audio.shape}")
    audio_data = predict_format(audio)
    print(f"Loaded and formatted audio. Audio shape: {audio_data.shape}")

    # Predict for each bird
    for bird in bird_choice:
        st.markdown(f"### 🐤 {bird.capitalize()}")

        # Initialize the model
        model = model.BirdNet(
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            lstm_hidden_units=lstm_hidden_units,
            filter_size=filter_size,
            name=f'{bird}filter'
        )

        # Compile the model and load weights
        model.build(audio_data.shape)
        weights_path = os.path.join(mpr, WEIGHTS_DIR, f"{bird}.h5")
        model.load_weights(weights_path)

        print(f"Loaded model with {weights_path}")

        with st.spinner(f"Analyzing for {bird}..."):
            probabilities = model.predict(audio_data)
            probabilities = 1 / (1 + np.exp(-probabilities))

        time_axis = np.arange(len(probabilities[0]))

        y_min = max(0, np.min(probabilities) - 0.00010)
        y_max = min(1, np.max(probabilities) + 0.00010)

        fig, ax = plt.subplots()
        ax.plot(time_axis, probabilities[0], label=f"{bird} Probability", color='green')
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Probability")
        ax.set_title(f"Probability of {bird.capitalize()} over Time")
        ax.set_ylim([y_min, y_max])
        ax.grid(True)
        st.pyplot(fig)

