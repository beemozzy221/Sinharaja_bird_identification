import numpy as np
import os
from os.path import join as pjoin
import waveletdecomp
import birdfileannotate

species_name = "Black-capped Bulbul"

dir_name = os.path.dirname(__file__)
data_dir = pjoin(dir_name, 'tests', species_name)
data_save = pjoin(dir_name, 'birddataset', species_name,f'{species_name}_targets.npy')

bird_targets = []

for file_name in os.listdir(data_dir):
    if file_name.endswith(".wav"):

        bird_targets.append(birdfileannotate.split_and_play_wav(pjoin(dir_name, 'tests', f'{species_name}' ,file_name)))

#Flatten the list
bird_targets = [annotation_values for sublist in bird_targets for annotation_values in sublist]

#Linearize output
bird_targets = birdfileannotate.linearizeoutput(np.array(bird_targets))

#Save the file as numpy
waveletdecomp.savenumpy(data_save, bird_targets)

