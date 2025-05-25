import os
import keras
import numpy as np
import callback_metric
from os.path import join as pjoin
from matplotlib import pyplot as plt
from recall_loss import custom_loss as binary_loss
from model import BirdNet

species_name = "SLBM"

dir_name = os.path.dirname(__file__)
model_save = pjoin(dir_name, "weights")
data_features = pjoin(dir_name, 'numpy_features', 'numpy_features.npy')
data_targets= pjoin(dir_name, 'numpy_targets', species_name, f'{species_name}_targets.npy')
bird_features = np.load(data_features)
bird_targets = np.load(data_targets)
dropout_rate = 0.1
learning_rate = 0.001
hidden_units = [256, 256]
lstm_hidden_units = [256,256]
filter_size = [32, 32]
epochs = 10
batch_size = 2


early_stopping = keras.callbacks.EarlyStopping(
        monitor="acc", patience=5, restore_best_weights=True
)

#Initialize the model
model = BirdNet(
    hidden_units = hidden_units,
    lstm_hidden_units = lstm_hidden_units,
    dropout_rate = dropout_rate,
    filter_size=filter_size,
    name="target_bird_filter"
)

#Compile the model
model.compile(
        optimizer=keras.optimizers.Adam(learning_rate= learning_rate),
        loss=binary_loss(0.7, 0.3, from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), "true_positives", "true_negatives", "false_positives", "false_negatives"]
)

#Train the model
metrics_logger = callback_metric.MetricsLogger()
history = model.fit(
        shuffle=True,
        x=bird_features,
        y=bird_targets,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, metrics_logger],
)

#Save model weights
model.save(f"{model_save}/{species_name}model.weights.h5")

#Plot TP, FP, FN, TN
epochs = range(1, len(metrics_logger.tp) + 1)

plt.plot(epochs, metrics_logger.tp, label='True Positives')
plt.plot(epochs, metrics_logger.fp, label='False Positives')
plt.plot(epochs, metrics_logger.tn, label='True Negatives')
plt.plot(epochs, metrics_logger.fn, label='False Negatives')

plt.title('Training Metrics Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()

# Access loss and accuracy
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history.get('acc')
val_acc = history.history.get('val_acc')

# Plot loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy (if available)
if train_acc and val_acc:
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()

# Write metrics to file
with open("training_metrics.txt", "w") as f:
    f.write("Epoch\tTP\tFP\tTN\tFN\tTrain_Loss\tVal_Loss\tTrain_Acc\tVal_Acc\n")
    for i in range(len(epochs)):
        f.write(f"{epochs[i]}\t"
                f"{metrics_logger.tp[i]}\t"
                f"{metrics_logger.fp[i]}\t"
                f"{metrics_logger.tn[i]}\t"
                f"{metrics_logger.fn[i]}\t"
                f"{train_loss[i]:.4f}\t"
                f"{val_loss[i]:.4f}\t"
                f"{(train_acc[i] if train_acc else 'NA')}\t"
                f"{(val_acc[i] if val_acc else 'NA')}\n")
