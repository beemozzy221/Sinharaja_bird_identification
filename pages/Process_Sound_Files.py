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
dropout_rate = 0.1
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
        weights_path = os.path.join("weights_npy", bird)

        # Initialize the model
        model = model.BirdNet(
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            lstm_hidden_units=lstm_hidden_units,
            filter_size=filter_size,
            name=f'{bird}filter'
        )

        weights_dict = {}
        for files in sorted(os.listdir(weights_path)):
            weights_dict[files] = []
            for trained_model in sorted(os.listdir(os.path.join(weights_path, files))):
                loaded_pieces = np.load(os.path.join(weights_path, files, trained_model))
                weights_dict[files].append(loaded_pieces)

        # Compile the model and load weights
        dummy_input = np.random.random((1, 203, 32, 1437, 1))
        model(dummy_input)

        for layer in model.layers:
            new_weights = []

            for weights in layer.weights:
                print(f"{layer.name} | {weights.name}")
                if weights.name == "kernel":
                    new_weights.append(weights_dict[layer.name][1])
                    print(f"Expected: {weights.shape} Received: {weights_dict[layer.name][1].shape}")
                    weights_dict[layer.name].pop(1)
                elif weights.name == "bias":
                    new_weights.append(weights_dict[layer.name][0])
                    print(f"Expected: {weights.shape} Received: {weights_dict[layer.name][0].shape}")
                    weights_dict[layer.name].pop(0)
                elif weights.name == "recurrent_kernel":
                    new_weights.append(weights_dict[layer.name][1])
                    print(f"Expected: {weights.shape} Received: {weights_dict[layer.name][1].shape}")
                    weights_dict[layer.name].pop(1)

            # Set the weights
            layer.set_weights(new_weights)

        # Get summary
        model.summary()
        print(f"Loaded model with {weights_path}")

        with st.spinner(f"Analyzing for {bird}..."):
            if audio_data.shape[3] > dummy_input.shape[3]:
                audio_data = audio_data[:, :, :, :dummy_input.shape[3], :]

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

