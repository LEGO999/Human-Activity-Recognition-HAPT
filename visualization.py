"""
This file is used to visualize the result for a whole sequence from the test set.
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import make_dataset

COLORMAP = ['w', 'grey', 'lightcoral', 'r', 'chocolate', 'sandybrown', 'gold', 'olive', 'g', 'royalblue',
            'plum', 'm', 'hotpink']
CLASS_NAME = ['UNLABELED', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
              'STAND_TO_SIT', 'SIT_TO_STAND', ' SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
              'LIE_TO_STAND']


def show(index, label, inputs, pred, image_path):
    plt.figure(figsize=(10, 10))
    index = list(index)
    inputs = list(inputs)
    acc_01 = [x[0] for x in inputs]
    acc_02 = [x[1] for x in inputs]
    acc_03 = [x[2] for x in inputs]
    gyro_01 = [x[3] for x in inputs]
    gyro_02 = [x[4] for x in inputs]
    gyro_03 = [x[5] for x in inputs]

    plt.subplot(5, 1, 1)
    plt.title('Accelorometer')
    plt.plot(index, acc_01, 'b')
    plt.plot(index, acc_02, 'g')
    plt.plot(index, acc_03, 'r')

    plt.subplot(5, 1, 2)
    plt.title('Gyroscope')
    plt.plot(index, gyro_01, 'b')
    plt.plot(index, gyro_02, 'g')
    plt.plot(index, gyro_03, 'r')

    plt.subplot(5, 1, 3)
    plt.title('Label')
    label = list(label)
    color_list = list()
    for i in range(len(label)):
        label_index = label[i]
        color = COLORMAP[label_index]
        color_list.append(color)
    # print(color_list)
    plt.vlines(index, 0, 5, linewidth=3, color=color_list)

    plt.subplot(5, 1, 4)
    plt.title('Prediction')
    pred = list(pred)
    color_list = list()
    for i in range(len(pred)):
        pred_index = pred[i]
        color = COLORMAP[pred_index]
        color_list.append(color)
    # print(color_list)
    plt.vlines(index, 0, 5, linewidth=3, color=color_list)

    plt.subplot(5, 1, 5)
    plt.title('Colormap')
    plt.bar(range(13), 5, color=COLORMAP)
    plt.xticks(range(13), CLASS_NAME, rotation=75)

    plt.subplots_adjust(hspace=1)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()


if __name__ == '__main__':
    show()