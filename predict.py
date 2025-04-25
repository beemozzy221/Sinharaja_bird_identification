import os
import numpy as np
import model
import waveletdecomp
from os.path import join as pjoin
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

filter_species_name = "SLBM"

dir_name = os.path.dirname(__file__)
data_dir = pjoin(dir_name, 'predict')
model_load = pjoin(dir_name, "weights", f'{filter_species_name}model.weights.h5')
dropout_rate = 0.1
hidden_units = [128, 128]
lstm_hidden_units = [128,128]
filter_size = [32, 32]
packets_to_display = 10

predict_data = []
predict_energy_data = []

for file_name in os.listdir(data_dir):
    if file_name.endswith((".WAV", ".wav")):
        print(f"Found {file_name}")

        file_path = os.path.join(data_dir, file_name)

        # Segment the file
        data = waveletdecomp.waveletsegment(file_path)

        #For each second
        for i in range(0, data.shape[0]):
            # Decompose the wav file
            packet = waveletdecomp.wavpacketdecomp(data[i])

            #Collect the coefficients
            _, nodes, energy = waveletdecomp.collect_coefficients(packet)

            #Append to list
            predict_data.append(nodes)
            predict_energy_data.append(energy)

predict_data = np.array(predict_data)
predict_data = predict_data.reshape(1, *predict_data.shape, 1)

predict_energy_data = np.array(predict_energy_data)
predict_energy_data = predict_energy_data.reshape(1, *predict_energy_data.shape, 1)

#Initialize the model
model = model.BirdNet(
    hidden_units = hidden_units,
    dropout_rate = dropout_rate,
    lstm_hidden_units = lstm_hidden_units,
    filter_size=filter_size,
    name=f'{filter_species_name}filter'
)

#Compile the model and load weights
model.build(predict_data.shape)
model.load_weights(model_load)

#Predict
results = model.predict(predict_data)

#Plot the results
x = np.linspace(1, results.shape[1], results.shape[1])
results = 1 / (1 + np.exp(-results))

x_smooth = np.linspace(x.min(), x.max(), 500)
y_smooth = make_interp_spline(x, results[0], k=2)(x_smooth)

'''plt.plot(x_smooth, y_smooth, label = "Prediction for each second")
plt.xlabel("Time")
plt.ylabel("Predictions")
plt.title("Smoothed predictions over time")
plt.legend()
plt.show()'''

#Plot the energy curves
xe = np.linspace(1, predict_energy_data.shape[1], predict_energy_data.shape[1])
ye = predict_energy_data[0]

xe_smooth = np.linspace(xe.min(), xe.max(), 500)
ye_smooth = make_interp_spline(xe, ye, k=2)(xe_smooth)

plt.subplot(1,2,1)
plt.plot(x_smooth, y_smooth, label = "Predicted values for each second")
plt.xlabel("Time")
plt.ylabel("Predictions")
plt.title("Smoothed predictions over time")
plt.legend()

plt.subplot(1,2,2)
for i in range(packets_to_display):
    plt.plot(xe_smooth, ye_smooth[:, i], label=f"Packet {i+1}")

plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Smoothed energy over time")
plt.legend()
plt.show()

#Identification
for time, prob in enumerate(results[0]):
    print(f"Present in {time}") if prob > 0.8 else print("None")