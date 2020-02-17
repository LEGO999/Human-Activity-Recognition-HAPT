"""
This file is ConfusionMatrix metrics for training
"""
import tensorflow as tf


class ConfusionMatrix(tf.metrics.Metric):

    def __init__(self, num_class):
        super().__init__()
        self.num = num_class
        self.weight = tf.Variable(initial_value=tf.zeros(shape=(num_class, num_class)), trainable=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones_like(y_pred)
        update = tf.math.confusion_matrix(y_true, y_pred, num_classes=13, weights=sample_weight, dtype=tf.float32)
        self.weight.assign(self.weight + update)

    def result(self):
        return self.weight

    def reset_states(self):
        self.weight.assign(tf.zeros_like(self.weight))


def accuracy():
    return tf.keras.metrics.CategoricalAccuracy()


if __name__ == '__main__':

    # Test categorical code
    # a = ConfusionMatrix(num_class=6)
    # acc = Accuracy()
    # b = tf.random.stateless_categorical(
    #     tf.math.log([[0.2, 0.2, 0.2, 0.2, 0.2]]), 5, seed=[7, 17])
    # e = tf.random.stateless_categorical(
    #     tf.math.log([[0.2, 0.2, 0.2, 0.2, 0.2]]), 5, seed=[7, 16])
    # c = tf.constant(5, shape=(5), dtype=tf.int64)
    # b = tf.squeeze(b)
    # e = tf.squeeze(e)
    # minus = tf.constant(5, dtype=tf.int64)
    # sample_weight = tf.cast(tf.math.not_equal(c, 5), tf.int64)
    # print(b)
    # print(c)
    # print(e)
    # print(sample_weight)
    # a.update_state(b, e)
    # acc.update_state(b, e)
    # print(a.result())
    # print(acc.result())
    # a.update_state(b,c)
    # acc.update_state(b, c, sample_weight=sample_weight)
    # print(a.result())
    # print(acc.result())

    a = ConfusionMatrix(num_class=5)
    acc = accuracy()

    la = tf.random.stateless_categorical(
        tf.math.log([[0.2, 0.2, 0.2, 0.2, 0.2]]), 5, seed=[7, 14])

    pre = tf.random.uniform(shape=(5, 4), minval=0., maxval=1., seed=7)
    pre = tf.nn.softmax(pre)
    pre_sparse = tf.argmax(pre, axis=1)

    la = tf.squeeze(la) - 1
    sample_weight = tf.cast(tf.math.not_equal(la, -1), tf.int64)
    print(la)
    print(pre_sparse)
    la = tf.one_hot(la, depth=4)
    pre = tf.squeeze(pre)
    minus = tf.constant(-1, dtype=tf.int64)
    print(sample_weight)
    # print(c)

    # print(sample_weight)
    acc.update_state(la, pre, sample_weight=sample_weight)
    print(acc.result())
    # a.update_state(b, e)
    # acc.update_state(b, e)
    # print(a.result())
    # print(acc.result())
    # a.update_state(b,c)
    # acc.update_state(b, c, sample_weight=sample_weight)
    # print(a.result())
    # print(acc.result())




