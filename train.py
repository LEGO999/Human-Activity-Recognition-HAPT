"""
This file is training routline for HAPT dataset
"""

import datetime
import os
import tensorflow as tf
from absl import logging
from time import time
from dataset import make_dataset
from metrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
from model import Lstm
import random
from visualization import show

logging.set_verbosity(logging.INFO)


# Define loss function
def loss(model, x, y, training):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    y_pred = model(x, training=training)
    return loss_object(y, y_pred)


# Define gradient
def grad(model, inputs, targets, training=True):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=training)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def plot_confusion_matrix(cm, class_names):
    """
    The function is to return a  matplotlib figure containing the plotted confusion matrix.
    Args:
        cm(array, shape = [n, n]): a confusion matrix of integer classes
        class_names(array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(7, 7))
    cm = cm.numpy()

    # Drop the zero column and row.
    cm = cm[1:, 1:]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=75)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix

    cm = np.around(cm.astype('float')/(cm.sum(axis=1))[:, np.newaxis], decimals=2)
    threshold = cm.max() / 1.3
    # Use white text if squares are dark; otherwise black
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    plt.ylim([11.25, -0.75])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
    The supplied figure is closed and inaccessible after the call
    """
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


# Define training function
def train(unit, dropout, learning_rate, num_epoch, tuning=True):

    num_epoch = int(num_epoch)
    log_dir = './results/'

    # Load dataset
    path = os.getcwd()
    train_file = path + '/hapt_tfrecords/hapt_train.tfrecords'
    val_file = path + '/hapt_tfrecords/hapt_val.tfrecords'
    test_file = path + '/hapt_tfrecords/hapt_test.tfrecords'

    train_dataset = make_dataset(train_file, overlap=True)
    val_dataset = make_dataset(val_file)
    test_dataset = make_dataset(test_file)
    class_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
                   'STAND_TO_SIT', 'SIT_TO_STAND', ' SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
                   'LIE_TO_STAND']

    # set a random batch number to visualize the result in test dataset.
    len_test = len(list(test_dataset))
    show_index = random.randint(10, len_test)

    # Model
    model = Lstm(unit=unit, drop_out=dropout)

    # set optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # set Metrics
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    val_accuracy = tf.keras.metrics.Accuracy()
    val_con_mat = ConfusionMatrix(num_class=13)
    test_accuracy = tf.keras.metrics.Accuracy()
    test_con_mat = ConfusionMatrix(num_class=13)

    # Save Checkpoint
    if not tuning:
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=5)

    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = log_dir + current_time
    summary_writer = tf.summary.create_file_writer(tb_log_dir)

    # Restore Checkpoint
    if not tuning:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logging.info('Restored from {}'.format(manager.latest_checkpoint))
        else:
            logging.info('Initializing from scratch.')

    # calculate losses, update network and metrics.
    @tf.function
    def train_step(inputs, label):
        # Optimize the model
        loss_value, grads = grad(model, inputs, label)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_pred = model(inputs, training=True)
        train_pred = tf.squeeze(train_pred)
        label = tf.squeeze(label)
        train_accuracy.update_state(label, train_pred, sample_weight=sample_weight)

    for epoch in range(num_epoch):
        begin = time()

        # Training loop
        for exp_num, index, label, train_inputs in train_dataset:
            train_inputs = tf.expand_dims(train_inputs, axis=0)
            # One-hot coding is applied.
            label = label - 1
            sample_weight = tf.cast(tf.math.not_equal(label, -1), tf.int64)
            label = tf.expand_dims(tf.one_hot(label, depth=12), axis=0)
            train_step(train_inputs, label)

        for exp_num, index, label, val_inputs in val_dataset:
            val_inputs = tf.expand_dims(val_inputs, axis=0)
            sample_weight = tf.cast(tf.math.not_equal(label, tf.constant(0, dtype=tf.int64)), tf.int64)
            val_pred = model(val_inputs, training=False)
            val_pred = tf.squeeze(val_pred)
            val_pred = tf.cast(tf.argmax(val_pred, axis=1), dtype=tf.int64) + 1
            val_con_mat.update_state(label, val_pred, sample_weight=sample_weight)
            val_accuracy.update_state(label, val_pred, sample_weight=sample_weight)
        # Log the confusion matrix as an image summary
        cm_valid = val_con_mat.result()
        figure = plot_confusion_matrix(cm_valid, class_names=class_names)
        cm_valid_image = plot_to_image(figure)

        with summary_writer.as_default():
            tf.summary.scalar('Train Accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('Valid Accuracy', val_accuracy.result(), step=epoch)
            tf.summary.image('Valid ConfusionMatrix', cm_valid_image, step=epoch)
        end = time()
        logging.info("Epoch {:d} Training Accuracy: {:.3%} Validation Accuracy: {:.3%} Time:{:.5}s".format(epoch + 1,
                     train_accuracy.result(), val_accuracy.result(), (end - begin)))

        train_accuracy.reset_states()
        val_accuracy.reset_states()
        val_con_mat.reset_states()

        if not tuning:
            if int(ckpt.step) % 5 == 0:
                save_path = manager.save()
                logging.info('Saved checkpoint for epoch {}: {}'.format(int(ckpt.step), save_path))
            ckpt.step.assign_add(1)

    i = 0
    for exp_num, index, label, test_inputs in test_dataset:
        test_inputs = tf.expand_dims(test_inputs, axis=0)
        sample_weight = tf.cast(tf.math.not_equal(label, tf.constant(0, dtype=tf.int64)), tf.int64)
        test_pred = model(test_inputs, training=False)
        test_pred = tf.cast(tf.argmax(test_pred, axis=2), dtype=tf.int64)
        test_pred = tf.squeeze(test_pred, axis=0) + 1
        test_accuracy.update_state(label, test_pred, sample_weight=sample_weight)
        test_con_mat.update_state(label, test_pred, sample_weight=sample_weight)
        i += 1

        # visualize the result
        if i == show_index:
            if not tuning:
                visualization_path = path + '/visualization/'
                image_path = visualization_path + current_time + '.png'
                inputs = tf.squeeze(test_inputs)
                show(index, label, inputs, test_pred, image_path)

    # Log the confusion matrix as an image summary
    cm_test = test_con_mat.result()
    figure = plot_confusion_matrix(cm_test, class_names=class_names)
    cm_test_image = plot_to_image(figure)

    with summary_writer.as_default():
        tf.summary.scalar('Test Accuracy', test_accuracy.result(), step=epoch)
        tf.summary.image('Test ConfusionMatrix', cm_test_image, step=epoch)

    logging.info("Trained finished. Final Accuracy in test set: {:.3%}".format(test_accuracy.result()))

    return test_accuracy.result()

