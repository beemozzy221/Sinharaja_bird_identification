import os

import numpy as np
import pygame
import tempfile
from scipy.io import wavfile
from pydub import AudioSegment

def split_and_play_wav(wav_fname):
    """
    ChatGPT generated code for extracting snippets (Cool code, I know).
    Splits a WAV file into 1-second snippets, plays each snippet, and collects user input (1/0).

    Args:
        wav_fname (str): Path to the WAV file.
    """
    #Read wav file
    samplerate, data = wavfile.read(wav_fname)

    # Load the audio file
    audio = AudioSegment.from_wav(wav_fname)

    #Snippet duration
    snippet_duration_ms = 1000

    # Calculate the number of snippets
    num_snippets = data.shape[0] // samplerate

    user_responses = []

    print("Playing audio snippets. Press 1 or 0 after each snippet.")

    for i in range(num_snippets):
        # Extract the snippet
        snippet = audio[i * snippet_duration_ms:(i + 1) * snippet_duration_ms]

        # Save snippet to a temporary file for playback
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            snippet.export(temp_file.name, format="wav")
            temp_path = temp_file.name

        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        # Wait for the snippet to finish playing
        while pygame.mixer.music.get_busy():
            pass

        # Get user input
        while True:
            user_input = input(f"Snippet {i + 1}/{num_snippets} - Enter 1 or 0: ")
            if user_input in ["1", "0"]:
                user_responses.append(int(user_input))
                break
            else:
                print("Invalid input. Please enter 1 or 0.")

        # Stop the mixer and delete the temporary file
        pygame.mixer.quit()
        os.remove(temp_path)

    print("Finished processing all snippets.")
    print("User responses:", user_responses)

    return user_responses

def linearizeoutput(data):
    return data.reshape(-1,1)