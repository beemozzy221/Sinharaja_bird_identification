import keras
from keras.src import ops
from keras.src.layers import Dense, Dropout, LSTM, Flatten, Conv2D, AveragePooling2D


def feed_forward (hidden_units,dropout_rate):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(Dense(units, activation="relu"))
        fnn_layers.append(Dropout(dropout_rate))

    return keras.Sequential(fnn_layers)

def lstm_layer (hidden_lstm_units):
    lstm_layers = []
    for units in hidden_lstm_units:
        lstm_layers.append(LSTM(units, return_sequences = True))

    return keras.Sequential(lstm_layers)

def conv_layer (filter_sizes):
    conv_layers = []
    for filter_size in filter_sizes:
        conv_layers.append(Conv2D(filters=filter_size, kernel_size=3,
                                  activation='relu', padding='same'))
        AveragePooling2D(pool_size=(2, 2),
                         strides=(1, 1), padding="same")

    return keras.Sequential(conv_layers)

class BirdNet(keras.Model):
    def __init__(self, hidden_units, dropout_rate, lstm_hidden_units, filter_size, *args,**kwargs):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.filter_sizes = filter_size
        self.lstm_hidden_units = lstm_hidden_units
        self.node_selector = Dense(1, activation="sigmoid")
        self.lstm_layers = lstm_layer(lstm_hidden_units)
        self.flatten = Flatten()
        self.postprocessor = feed_forward(hidden_units, dropout_rate)
        self.conv = conv_layer(filter_size)
        self.compute_logits = Dense(1)
        self.bs_dense = Dense(64)

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size, time_steps, features, coeffs, channel = (input_shape[0], input_shape[1], input_shape[2],
                                                     input_shape[3], input_shape[4])

        x0 = ops.reshape(inputs, (batch_size * time_steps, features, coeffs, channel))
        x1 = self.conv(x0)

        conv_output = ops.shape(x1)
        conv_features = conv_output[3]

        node_inputs = ops.reshape(x1, (batch_size, time_steps, features,
                                       coeffs * conv_features))

        x2 = self.node_selector(node_inputs)
        x3 = ops.reshape(x2, (batch_size, time_steps, features))
        x4 = self.lstm_layers(x3)

        return self.compute_logits(x4)

    def build(self, input_shape):
        super(BirdNet, self).build(input_shape)

    def get_config(self):
        return {"hidden_units":self.hidden_units,
                "lstm_hidden_units": self.lstm_hidden_units,
                "dropout_rate":self.dropout_rate,
                "filter_sizes": self.filter_sizes}

