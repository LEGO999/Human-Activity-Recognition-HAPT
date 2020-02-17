"""
This file is used to make dataset for training routine
"""

import tensorflow as tf
import os


def make_dataset(file, overlap=False):
    raw_dataset = tf.data.TFRecordDataset(file)

    # Create a dictionary describing the features
    feature_description = {
        'exp_num': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'index': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'acc_01': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'acc_02': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'acc_03': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'gyro_01': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'gyro_02': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'gyro_03': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        dataset = tf.io.parse_single_example(example_proto, feature_description)
        exp_num = dataset['exp_num']
        index = dataset['index']
        label = dataset['label']
        acc_01 = dataset['acc_01']
        acc_02 = dataset['acc_02']
        acc_03 = dataset['acc_03']
        gyro_01 = dataset['gyro_01']
        gyro_02 = dataset['gyro_02']
        gyro_03 = dataset['gyro_03']
        inputs = tf.stack([acc_01, acc_02, acc_03, gyro_01, gyro_02, gyro_03])
        return exp_num, index, label, inputs

    dataset = raw_dataset.map(_parse_function)

    # sliding window
    if overlap:
        dataset = dataset.window(250, shift=125).flat_map(lambda a, b, c, d: zip(a, b, c, d)).batch(250)
    if not overlap:
        dataset = dataset.batch(250)
    return dataset


if __name__ == '__main__':
    path = os.getcwd()
    val_file = path + '/HAPTcsv/hapt_train.tfrecords'
    val_dataset = make_dataset(val_file, overlap=True)
    for exp_num, index, label, inputs in val_dataset:
        print(label-1)







