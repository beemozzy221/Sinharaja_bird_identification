import os
import waveletdecomp
import numpy as np
from os.path import join as pjoin

species_name = "Black-capped Bulbul"
dir_name = os.path.dirname(__file__)
data_dir = pjoin(dir_name, 'tests', species_name)
data_save = pjoin(dir_name, 'birddataset', species_name,f'{species_name}_dataset.npy')

#Initiate a list for collecting the nodes
nodes_data = []

# List all files in the folder
for file_name in os.listdir(data_dir):
    if file_name.endswith(".wav"):
        print(f"Found {file_name}")

        file_path = os.path.join(data_dir, file_name)

        # Segment the file
        data = waveletdecomp.waveletsegment(file_path)

        #For each second
        for i in range(0, data.shape[0]):
            # Decompose the wav file
            packet = waveletdecomp.wavpacketdecomp(data[i])

            #Collect the coefficients
            _, _, nodes = waveletdecomp.collect_coefficients(packet)

            #Append numpy
            nodes_data.append(nodes)

nodes_data = np.array(nodes_data)
nodes_data = nodes_data.reshape(*nodes_data.shape, 1)

#Save the dataset
#waveletdecomp.savenumpy(data_save, nodes_data)


