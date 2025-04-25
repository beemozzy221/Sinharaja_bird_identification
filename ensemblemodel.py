import os
import keras
import numpy as np
import model
import waveletdecomp
import tensorflow as tf
from os.path import join as pjoin
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

filter_species_name = "Black-capped Bulbul"

dir_name = os.path.dirname(__file__)
predict_dir = pjoin(dir_name, 'predict')
model_dir = pjoin(dir_name, 'weights')
dropout_rate = 0.1
hidden_units = [128, 128]
lstm_hidden_units = [128,128]
packets_to_display = 10
no_of_species = 1

#Load to-predict data
predict_data = []

for file_name in os.listdir(predict_dir):
    if file_name.endswith(".wav"):
        print(f"Found {file_name}")

        file_path = os.path.join(predict_dir, file_name)

        # Segment the file
        data = waveletdecomp.waveletsegment(file_path)

        #For each second
        for i in range(0, data.shape[0]):
            # Decompose the wav file
            packet = waveletdecomp.wavpacketdecomp(data[i])

            #Collect the coefficients
            _, _, nodes = waveletdecomp.collect_coefficients(packet)

            #Append to list
            predict_data.append(nodes)

predict_data = np.array(predict_data)
predict_data = predict_data.reshape(*predict_data.shape, 1)

model_data = []

for file_name in os.listdir(model_dir):
    if file_name.endswith(".weights.h5"):
        print(f"Found {file_name}")

        file_path = os.path.join(model_dir, file_name)

        #Append each weights
        model_data.append(file_path)

#Initialize the model
model = model.BirdNet(
    hidden_units = hidden_units,
    dropout_rate = dropout_rate,
    lstm_hidden_units = lstm_hidden_units,
    name=f'aggregated_bird_filter'
)

#Compile the model and load weights
model.build(predict_data.shape)

# Example list of model objects (these can be pre-built or loaded models)
models = [keras.models.clone_model(model) for _ in range(len(model_data))]

# Load weights into each model using a loop
for model, weight_file in zip(models, model_data):
    model.load_weights(weight_file)
    print(f"Loaded weights from {weight_file} into model.")

#Ennsemble model
model_input = keras.Input(shape=predict_data.shape[1:])
merged = keras.layers.Concatenate()([model(model_input) for model in models])
ensemble_output = keras.layers.Softmax()(merged)
ensemble_model = keras.Model(inputs=model_input, outputs=ensemble_output)

#Predict
results = ensemble_model.predict(predict_data)

#Plot results
time_steps = np.linspace(0, results.shape[0], results.shape[0])
time_steps_smooth = np.linspace(time_steps.min(), time_steps.max(), 500)

for classes in range(results.shape[1]):
    predictions_smooth = make_interp_spline(time_steps, results[:, classes], k=2)(time_steps_smooth)
    plt.plot(time_steps_smooth, predictions_smooth, label=f'Class {classes}')

plt.xlabel("Timesteps")
plt.ylabel("Predictions")
plt.title("Softmax probabilities for each class")
plt.legend()
plt.show()