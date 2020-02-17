"""
This file is the main function
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import gin.tf
from train import train
import tensorflow as tf
from bayes_opt import BayesianOptimization


@gin.configurable
def main(unit, dropout, learning_rate, num_epoch, tuning):
    if tuning:
        hyperparameters = {'unit': (16, 64), 'dropout': (0.0, 0.4), 'learning_rate': (1e-2, 1e-5), 'num_epoch': (num_epoch, num_epoch)}
        optimizer = BayesianOptimization(
            f=train,
            pbounds=hyperparameters,
            verbose=2,
            random_state=1,
        )
        optimizer.maximize(init_points=5, n_iter=5)
        print(optimizer.max)
    else:
        train(unit=unit, dropout=dropout, learning_rate=learning_rate, num_epoch=num_epoch, tuning=tuning)


if __name__ == "__main__":
    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    gin.parse_config_file('config.gin')
    main()

