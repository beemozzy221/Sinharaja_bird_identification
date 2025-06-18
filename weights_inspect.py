import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from model import BirdNet

print(tf.version.VERSION)

dropout_rate = 0.1
hidden_units = [512, 512]
lstm_hidden_units = [128, 128]
filter_size = [32, 32, 32]

weights_path = 'weights_npy/SLBM'

# List only files (not directories)
weights_dict = {}

#Load the weights
for files in sorted(os.listdir(weights_path)):
    weights_dict[files] = []
    for trained_model in sorted(os.listdir(os.path.join(weights_path, files))):
        loaded_pieces = np.load(os.path.join(weights_path, files, trained_model))
        weights_dict[files].append(loaded_pieces)


# Open the HDF5 file
'''with h5py.File('weights/SLBMmodel.weights.h5', 'r') as f:
    # List all groups and datasets
    def print_structure(name):
        print(name)
    #f.visit(print_structure)
    print(f['model_weights'].keys())'''

'''with h5py.File('weights/SLBMmodel.weights.h5', 'r') as f:
    # Top-level groups are typically 'model_weights' or just layer names
    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name} | shape: {obj.shape}")
    f.visititems(visit_func)'''

model = BirdNet(
    hidden_units = hidden_units,
    lstm_hidden_units = lstm_hidden_units,
    dropout_rate = dropout_rate,
    filter_size=filter_size,
    name="target_bird_filter"
)

dummy_input = np.load("numpy_features/numpy_features2.npy")
dummy_input = dummy_input[:1]

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

    #Set the weights
    layer.set_weights(new_weights)

# Get summary
model.summary()
#Test predict
results = model.predict(dummy_input)

plt.plot(range(results.shape[1]), tf.nn.sigmoid(results[-1]), label="Prediction plots")
plt.xlabel("Time frame")
plt.ylabel("Predictions")
plt.show()