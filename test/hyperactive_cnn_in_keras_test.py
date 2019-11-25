# coding: utf-8

# Example with a convolutional neural network in keras

import time
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from hyperactive import RandomSearchOptimizer, ParticleSwarmOptimizer

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# this defines the structure of the model and print("time: {}".format(t2-t1))the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["SGD"]},
    "keras.fit.0": {"epochs": [5], "batch_size": [500], "verbose": [2]},
    "keras.layers.Conv2D.1": {
        "filters": [32, 64, 128],
        #"kernel_size": range(3, 4),
        "kernel_size": [(3, 3)],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.3": {
        "filters": [16, 32, 64, 128],
        "kernel_size": [(3, 3)],
        "activation": ["relu"],
    },
    "keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
    "keras.layers.Flatten.5": {},
    #"keras.layers.Dense.6": {"units": [30], "activation": ["relu"]},
    "keras.layers.Dense.6": {"units": range(30, 100, 10), "activation": ["relu"]},
    #"keras.layers.Dropout.7": {"rate": 0.4},
    "keras.layers.Dropout.7": {"rate": list(np.arange(0.2, 0.8, 0.2))},
    "keras.layers.Dense.8": {"units": [10], "activation": ["softmax"]},
}

start_point = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [5], "batch_size": [500], "verbose": [0]},
    "keras.layers.Conv2D.1": {
        "filters": [64],
        "kernel_size": [3],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.3": {
        "filters": [32],
        "kernel_size": [3],
        "activation": ["relu"],
    },
    "keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
    "keras.layers.Flatten.5": {},
    "keras.layers.Dense.6": {"units": [30], "activation": ["relu"]},
    "keras.layers.Dropout.7": {"rate": [0.2]},
    "keras.layers.Dense.8": {"units": [10], "activation": ["softmax"]},
}

Optimizer = RandomSearchOptimizer(search_config, metric='accuracy', warim_start=start_point, n_iter=5)  # verbosity=1で最適パラメータ表示
#Optimizer = ParticleSwarmOptimizer(search_config, n_iter=20)

t1 = time.time()
# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

t2 = time.time()
print("time: {}".format(t2-t1))

# predict from test data
Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)

print("test score: {}".format(score))