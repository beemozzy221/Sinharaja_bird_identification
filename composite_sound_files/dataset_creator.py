import math
import os
import waveletdecomp
import numpy as np
from pydub import AudioSegment
from pandas import read_csv
from natsort import natsorted
from os.path import join as pjoin
from sklearn.preprocessing import MultiLabelBinarizer

def fix_audio_length_pydub(parent_sound_file_path, max_length):
    for recording_no in natsorted(os.listdir(parent_sound_file_path)):
        for recording_file in natsorted(os.listdir(pjoin(parent_sound_file_path, recording_no))):

            audio = AudioSegment.from_file(pjoin(parent_sound_file_path, recording_no, recording_file))
            target_length = max_length + 1

            if len(audio) < target_length:
                silence = AudioSegment.silent(duration=target_length - len(audio))
                fixed = audio + silence
                fixed = fixed[:target_length]

            else:
                fixed = audio[:target_length]

            fixed.export(pjoin(parent_sound_file_path, recording_no, recording_file), format="wav")

def birdgroup_maxlen(parent_sound_file_path):
    length_data = []

    # List all files in the parent recording folder
    for recording_main in natsorted(os.listdir(parent_sound_file_path)):
        for recording_file in natsorted(os.listdir(pjoin(parent_sound_file_path, recording_main))):
            if recording_file.endswith((".WAV", ".wav")):
                print(f"Found {recording_file} in {parent_sound_file_path}")

                # Find length
                length = waveletdecomp.waveletinfo(pjoin(parent_sound_file_path, recording_main, recording_file))

                length_data.append(length)

    return max(length_data)

def create_dataset_for_segment(parent_sound_file_path):
    """args: accepts the folder path which has the recording in order.
    Make sure to only use this function when the wav files are of the exact same size, otherwise, during the conversion to numpy arrays,
    an inhomogeneous error will be thrown."""

    wav_decomposed = []

    # List all files in the parent recording folder
    for recording_no in natsorted(os.listdir(parent_sound_file_path)):
        for recording_file in natsorted(os.listdir(pjoin(parent_sound_file_path, recording_no))):
            if recording_file.endswith((".WAV", ".wav")):
                print(f"Found {recording_file} in {recording_no}")

                # Segment the file
                data = waveletdecomp.waveletsegment(pjoin(parent_sound_file_path, recording_no, recording_file))
                print(f"Data shape: {data.shape}")

                temp_node_data = []

                # For each second
                for i in range(0, data.shape[0]):

                    # Decompose the wav file
                    packet = waveletdecomp.wavpacketdecomp(data[i])

                    # Collect the coefficients
                    _, nodes, _ = waveletdecomp.collect_coefficients(packet)

                    # Append numpy
                    temp_node_data.append(nodes)

                wav_decomposed.append(temp_node_data)

    wav_decomposed = np.array(wav_decomposed)

    return wav_decomposed.reshape(*wav_decomposed.shape, 1)

def padding2d(dataarray, max_length):
    if dataarray.shape[0] == max_length:
        print("No need for paddings")

        return dataarray

    padding_size = max_length - dataarray.shape[0]
    padded_array = np.pad(dataarray,((0,padding_size),(0,0)), mode = "constant")
    print(f"Padded by {padding_size}s")

    return padded_array

def convert_annotatfiles_to_list(parent_sound_file_path, segment_length, dataset_filepath):
    recording_data = []

    #Import dataset
    dataset_df = read_csv(dataset_filepath)

    for recording_file in natsorted(os.listdir(parent_sound_file_path)):
        for recording_segment in natsorted(os.listdir(pjoin(parent_sound_file_path, recording_file))):
            if recording_segment.endswith((".WAV", ".wav")):
                annotation_segment = np.zeros(shape=segment_length, dtype="object")

                print(f"Opened {recording_segment} in {recording_file}")

                rec_name_and_seg = recording_segment.split('.')[0]

                #Isolate the appropriate rows
                if dataset_df[dataset_df["Record_Segment"] == rec_name_and_seg].empty:
                    recording_data.append(annotation_segment)
                    continue
                rows = dataset_df[dataset_df["Record_Segment"] == rec_name_and_seg]

                #Write for each segment
                for row in rows.iterrows():
                    for seconds in range(math.ceil(row[1]["Call_time"])):

                        if annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds] == 0:
                            annotation_segment[math.floor(row[1]["Begin Time (s)"] + seconds)] = row[1]["Species"]
                        elif row[1]["Species"] in annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds]:
                            continue
                        else:
                            annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds] += " " + row[1]["Species"]

                recording_data.append(annotation_segment)

    return recording_data

def convert_speci_species_column_to_onehot(target_specie, np_dataset):

    #Iterate through the numpy dataset
    annotated_numpy_encoded_shape = (*np_dataset.shape, 1)
    annotated_numpy_encoded = np.zeros(shape=annotated_numpy_encoded_shape, dtype="int")
    for segindex, segments in enumerate(np_dataset):
        for spindex, species in enumerate(segments):
            if species == 0:
                continue

            #Collect species list and transform
            species_lis = species.split(" ")
            annotated_numpy_encoded[segindex, spindex] = 1 if target_specie in species_lis else 0

    return annotated_numpy_encoded

def annotate_call_sounds_target_specie(parent_sound_file_path, segment_length, dataset_filepath, target_specie):
    recording_data = []

    #Import dataset
    dataset_df = read_csv(dataset_filepath)

    for recording_file in natsorted(os.listdir(parent_sound_file_path)):
        for recording_segment in natsorted(os.listdir(pjoin(parent_sound_file_path, recording_file))):
            annotation_segment = np.zeros(shape=segment_length, dtype="object")

            if recording_segment.endswith((".WAV", ".wav")):
                print(f"Opened {recording_segment} in {recording_file}")

                rec_name_and_seg = recording_segment.split('.')[0]

                #Isolate the appropriate rows
                if dataset_df[dataset_df["Record_Segment"] == rec_name_and_seg].empty:
                    recording_data.append(annotation_segment)
                    continue
                rows = dataset_df[dataset_df["Record_Segment"] == rec_name_and_seg]

                #Write for each segment
                for row in rows.iterrows():
                    if row[1]["Species"] == target_specie:
                        for seconds in range(math.ceil(row[1]["Call_time"])):

                            if annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds] == 0:
                                annotation_segment[math.floor(row[1]["Begin Time (s)"] + seconds)] = row[1]["Bird Sound"]
                            elif row[1]["Bird Sound"] in annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds]:
                                continue
                            else:
                                annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds] += " " + row[1]["Bird Sound"]

            recording_data.append(annotation_segment)

    return recording_data

def convert_vocal_column_to_onehot(df_dataset ,np_dataset):

    #Collect unique species
    fit_transform_array = df_dataset["Bird Sound"].unique()

    #Instantiate the one - hot encoder class and fit  # Define the category order
    encoder = MultiLabelBinarizer()
    encoder.fit(fit_transform_array.reshape(-1, 1))

    #Iterate through the numpy dataset
    annotated_numpy_encoded_shape = (*np_dataset.shape, fit_transform_array.shape[0])
    annotated_numpy_encoded = np.zeros(shape=annotated_numpy_encoded_shape, dtype="int")
    for segindex, segments in enumerate(np_dataset):
        for spindex, sound in enumerate(segments):
            if sound == 0:
                continue

            #Collect species list and transform
            sound_lis = np.array(sound.split(" "))
            sound_lis = sound_lis.reshape(1, -1)
            annotated_numpy_encoded[segindex, spindex] = encoder.transform(sound_lis)

    return annotated_numpy_encoded

if __name__ == "__main__":
    species_name = "SLBM"

    # Main file
    mpr = os.path.dirname(__file__)

    # Source fIles
    annot_files = pjoin(mpr, "Annotation", "Annotations.csv")
    parent_sound_file = pjoin(mpr, "Recordings")

    # Target files
    numpy_targets_file = pjoin(mpr, "numpy_targets", species_name, f"{species_name}_targets.npy")
    numpy_targets_vocal_file = pjoin(mpr, "numpy_targets", species_name, f"{species_name}_vocal_targets.npy")
    numpy_features_file = pjoin(mpr, "numpy_features", "numpy_features.npy")

    # Get max length
    max_length = birdgroup_maxlen(parent_sound_file)
    print(f"Maximum length of all audio files: {max_length}")

    # Pad all audio files
    #fix_audio_length_pydub(parent_sound_file, max_length * 1000)

    # Create features
    #numpy_features = create_dataset_for_segment(parent_sound_file)
    #print(numpy_features.shape)
    #waveletdecomp.savenumpy(numpy_features_file, numpy_features)

    # Create targets
    combined_targets_numpy_unencoded = np.array(convert_annotatfiles_to_list(parent_sound_file_path=parent_sound_file,
                                                                   segment_length=max_length,
                                                                   dataset_filepath=annot_files))
    specie_targets_numpy_encoded = convert_speci_species_column_to_onehot(species_name, combined_targets_numpy_unencoded)
    waveletdecomp.savenumpy(numpy_targets_file, specie_targets_numpy_encoded)
    print(specie_targets_numpy_encoded.shape)

    # Annotate based on call sounds for target specie
    #call_sound_unencoded = annotate_call_sounds_target_specie(parent_sound_file, max_length,
                                                              #annot_files, species_name)
    #call_sound_encoded = convert_vocal_column_to_onehot(read_csv(annot_files), np.array(call_sound_unencoded))
    #print(call_sound_encoded.shape)
    #waveletdecomp.savenumpy(numpy_targets_vocal_file, np.array(call_sound_encoded))


