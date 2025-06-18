import os
import numpy as np
import h5py


# Path to your saved weights file
weights_file = 'weights/SLBMmodel.weights.h5'

# Output folder for individual weight files
output_folder = 'weights_npy/SLBM'
os.makedirs(output_folder, exist_ok=True)

with h5py.File(weights_file, "r") as f:
    def recurse(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name)
    f.visititems(recurse)