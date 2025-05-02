import os
from pydub import AudioSegment
from os.path import join as pjoin
from pydub.utils import which

def convert_wav(parent_sound_file_path):
    print(which("ffmpeg"))
    AudioSegment.converter = which("ffmpeg")

    for recording_no in os.listdir(parent_sound_file_path):
        for recording_file in os.listdir(pjoin(parent_sound_file_path, recording_no)):

            if recording_file.endswith(".mp3"):
                # Load your MP3 file
                sound = AudioSegment.from_mp3(pjoin(parent_sound_file_path, recording_no, recording_file))

                # Export it as WAV
                export_name = os.path.splitext(recording_file)[0] + ".wav"
                sound.export(pjoin(parent_sound_file_path, recording_no, export_name), format="wav")

if __name__ == "__main__":
    # Main file
    mpr = os.path.dirname(__file__)

    # Source files
    parent_sound_file = pjoin(mpr, "Recordings")

    #Conversion
    convert_wav(parent_sound_file)
