import tensorflow as tf

class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.tp = []
        self.fp = []
        self.tn = []
        self.fn = []

    def on_epoch_end(self, epoch, logs=None):
        # Append metrics to the lists
        self.tp.append(logs.get('true_positives'))
        self.fp.append(logs.get('false_positives'))
        self.tn.append(logs.get('true_negatives'))
        self.fn.append(logs.get('false_negatives'))

        print(f" Epoch {epoch + 1}: TP={self.tp[-1]}, FP={self.fp[-1]}, TN={self.tn[-1]}, FN={self.fn[-1]}")