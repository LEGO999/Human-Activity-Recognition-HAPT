"""
This file is used as data preprocessing steps before input pipeline
"""

import os
import glob
import numpy as np
import pandas as pd
import shutil
import re
import matplotlib.pyplot as plt
from scipy.stats import zscore
import tensorflow as tf
from absl import logging

logging.set_verbosity(logging.INFO)


def to_csv(path):
    # change the txt file to csv and save in different folders
    txt_path = os.path.join(path, 'RawData/*.txt')
    txt_files = glob.glob(txt_path)
    for filename in txt_files:
        txt = np.loadtxt(filename)
        txtDF = pd.DataFrame(txt)
        name = os.path.splitext(filename)
        csv_file = name[0] + '.csv'
        txtDF.to_csv(csv_file, index=False)
        csv_new_file = csv_file.replace('RawData', 'RDcsv')
        shutil.move(csv_file, csv_new_file)
        logging.info('{} is created'.format(csv_new_file))


def label_transform(path):
    # show the signals and transfer labels to time series data.
    csv_path = path + '/RDcsv/*.csv'
    label_path = path + '/RDcsv/labels.csv'
    csv_files = glob.glob(csv_path)

    for filename in csv_files:
        exp = os.path.split(filename)[1]
        exp = exp.split('.')[0]
        sensor = exp.split('_')[0]

        if sensor == 'acc':
            exp_num = re.findall("\d+", exp)[0]
            user_num = re.findall("\d+", exp)[1]

            acc_file = filename
            gyro_file = acc_file.replace('acc', 'gyro')

            df_acc = pd.read_csv(acc_file)
            df_gyro = pd.read_csv(gyro_file)

            df_labels = pd.read_csv(label_path)
            label_filename = path + '/RDcsv/label_exp' + exp_num + '.csv'
            df_exp = df_labels[df_labels['0'] == int(exp_num)]
            raw = len(df_exp)
            last = df_exp.iloc[raw - 1, 4]
            df = pd.DataFrame(columns=['x', 'label'])
            for i in range(int(last) + 1):
                label = 0
                for index in range(raw):
                    if i >= df_exp.iloc[index, 3] and i <= df_exp.iloc[index, 4]:
                        label = int(df_exp.iloc[index, 2])
                df = df.append({'x': i, 'label': label}, ignore_index=True)

            df.to_csv(label_filename, encoding="gbk", index=False)

            logging.info('experiment {}, user {}'.format(exp_num, user_num))

            plt.subplot(2, 1, 1)
            plt.title('Accelorometer')
            plt.plot(df_acc.index, df_acc['0'], 'b')
            plt.plot(df_acc.index, df_acc['1'], 'g')
            plt.plot(df_acc.index, df_acc['2'], 'r')

            plt.subplot(2, 1, 2)
            plt.title('Gyroscope')
            plt.plot(df_gyro.index, df_gyro['0'], 'b')
            plt.plot(df_gyro.index, df_gyro['1'], 'g')
            plt.plot(df_gyro.index, df_gyro['2'], 'r')

            plt.show()


def input_normalization(path):
    # Input normalization and assign the label to the input signals
    csv_path = path + '/RDcsv/*.csv'
    csv_files = glob.glob(csv_path)

    for filename in csv_files:
        exp = os.path.split(filename)[1]
        exp = exp.split('.')[0]
        sensor = exp.split('_')[0]

        if sensor == 'acc':
            exp_num = re.findall("\d+", exp)[0]
            user_num = re.findall("\d+", exp)[1]

            acc_file = filename
            gyro_file = acc_file.replace('acc', 'gyro')
            label_file = path + '/RDcsv/label_exp' + exp_num + '.csv'

            df_acc = pd.read_csv(acc_file)
            df_gyro = pd.read_csv(gyro_file)
            df_label = pd.read_csv(label_file)

            df_acc = df_acc.apply(zscore)
            df_gyro = df_gyro.apply(zscore)

            df_acc = df_acc.rename(columns={'0': 'acc_01', '1': 'acc_02', '2': 'acc_03'})
            df_gyro = df_gyro.rename(columns={'0': 'gyro_01', '1': 'gyro_02', '2': 'gyro_03'})
            df = pd.concat([df_label, df_acc, df_gyro], axis=1, join_axes=[df_label.index])

            file_name = path + '/HAPTcsv/HAPT_exp' + exp_num + '_user' + user_num + '.csv'
            df.to_csv(file_name, encoding="gbk", index=False)
            logging.info('The input_label file {} for experiment {} is created'.format(file_name, exp_num))


def split_data(path):
    # Split the data
    csv_path = path + '/HAPTcsv'
    csv_filepath = csv_path + '/*.csv'
    train_path = csv_path + '/Train/'
    val_path = csv_path + '/Validation/'
    test_path = csv_path + '/Test/'

    csv_files = glob.glob(csv_filepath)
    for filename in csv_files:
        exp = os.path.split(filename)[1]
        exp = exp.split('.')[0]

        user_num = re.findall("\d+", exp)[1]
        user_num = int(user_num)

        if user_num in range(1, 22):
            train_file = train_path + exp + '.csv'
            shutil.move(filename, train_file)
            logging.info('{} is created'.format(train_file))
        elif user_num in range (22, 28):
            test_file = test_path + exp + '.csv'
            shutil.move(filename, test_file)
            logging.info('{} is created'.format(test_file))
        else:
            val_file = val_path + exp + '.csv'
            shutil.move(filename, val_file)
            logging.info('{} is created'.format(val_file))


# Create TFRecord
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate_tf_records(csv_path, name):
    csv_filepath = csv_path + '/*.csv'
    csv_files = glob.glob(csv_filepath)

    def serialize_example(exp_num, index, label, acc_01, acc_02, acc_03, gyro_01, gyro_02, gyro_03):
        # Create a tf.Example message ready to be written to a file
        feature = {
            'exp_num': _int64_feature(exp_num),
            'index': _int64_feature(index),
            'label': _int64_feature(label),
            'acc_01': _float_feature(acc_01),
            'acc_02': _float_feature(acc_02),
            'acc_03': _float_feature(acc_03),
            'gyro_01': _float_feature(gyro_01),
            'gyro_02': _float_feature(gyro_02),
            'gyro_03': _float_feature(gyro_03),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    record_file = path + '/HAPTcsv/hapt_' + name + '.tfrecords'

    with tf.io.TFRecordWriter(record_file) as writer:
        for filename in csv_files:
            logging.info('Generating{}'.format(filename))
            df = pd.read_csv(filename)
            exp = os.path.split(filename)[1]
            exp = exp.split('.')[0]
            exp_num = int(re.findall("\d+", exp)[0])

            for i in df.index:
                index = df.loc[i, 'x']
                label = df.loc[i, 'label']
                acc_01 = df.loc[i, 'acc_01']
                acc_02 = df.loc[i, 'acc_02']
                acc_03 = df.loc[i, 'acc_03']
                gyro_01 = df.loc[i, 'gyro_01']
                gyro_02 = df.loc[i, 'gyro_02']
                gyro_03 = df.loc[i, 'gyro_03']
                example = serialize_example(exp_num, index, label, acc_01, acc_02, acc_03, gyro_01, gyro_02, gyro_03)

                writer.write(example)
    writer.close()


if __name__ == '__main__':
    path = os.getcwd()
    # to_csv(path)
    # label_transform(path)
    # input_normalization(path)
    # split_data(path)
    train_path = path + '/HAPTcsv/Train/'
    val_path = path + '/HAPTcsv/Validation/'
    test_path = path + '/HAPTcsv/Test/'
    # generate_tf_records(train_path, name='train')
    generate_tf_records(val_path, name='val')
    # generate_tf_records(test_path, name='test')










