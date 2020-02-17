import tensorflow as tf


class Lstm(tf.keras.Model):

    def __init__(self, unit, drop_out):
        super(Lstm, self).__init__()
        self.unit = int(unit)
        self.gru0 = tf.keras.layers.GRU(self.unit * 2, dropout=drop_out, return_sequences=True)
        self.gru1 = tf.keras.layers.GRU(self.unit, dropout=drop_out, return_sequences=True)
        self.dense0 = tf.keras.layers.Dense(self.unit, activation='relu')
        self.dropout0 = tf.keras.layers.Dropout(drop_out)
        self.dense1 = tf.keras.layers.Dense(12, activation='softmax')

    def call(self, inputs, training=True):
        x = self.gru0(inputs)
        x = self.gru1(x)
        x = self.dense0(x)
        x = self.dropout0(x, training=training)
        x = self.dense1(x)
        return x


if __name__ == '__main__':
    lstm = Lstm()
    lstm.build(input_shape=(None, 250, 6))
    lstm.summary()