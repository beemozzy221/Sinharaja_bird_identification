import os
import numpy as np
import waveletdecomp
from os.path import join as pjoin


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


if __name__ == "__main__":

    #Encode the annotation file
    species_name = "EATO"
    annotated_save_unencoded = pjoin(os.path.dirname(__file__), "combined_numpy_dataset", "annotated_numpy_data_unencoded.npy")
    annotated_encoded = pjoin(os.path.dirname(__file__), "combined_numpy_dataset", species_name, "targets_encoded.npy")
    np_annotated_array = np.load(annotated_save_unencoded, allow_pickle=True)
    annotated_dataset_encoded = convert_speci_species_column_to_onehot("EATO", np_annotated_array)

    #Save the numpy
    waveletdecomp.savenumpy(annotated_encoded, annotated_dataset_encoded)