import tensorflow as tf


class ASR(tf.keras.Model):
    def __init__(self, filters, kernel_size, conv_stride, conv_border, n_lstm_units, n_dense_units):
        super(ASR, self).__init__()
        self.conv_layer = tf.keras.layers.Conv1D(filters,
                                                 kernel_size,
                                                 strides=conv_stride,
                                                 padding=conv_border,
                                                 activation='relu')
        self.lstm_layer = tf.keras.layers.LSTM(n_lstm_units,
                                               return_sequences=True,
                                               activation='tanh')
        self.lstm_layer_back = tf.keras.layers.LSTM(n_lstm_units,
                                                    return_sequences=True,
                                                    go_backwards=True,
                                                    activation='tanh')
        self.blstm_layer = tf.keras.layers.Bidirectional(
            self.lstm_layer, backward_layer=self.lstm_layer_back)
        self.dense_layer = tf.keras.layers.Dense(n_dense_units)

    def call(self, x):
        x = self.conv_layer(x)
        x = self.blstm_layer(x)
        x = self.dense_layer(x)
        return x

    def save(self, path):
        self.save_weights(path)

    def load(self, path):

        self.load_weights(path)
