# coding: utf-8

# Example with a convolutional neural network in keras

import time
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

from Hyperactive.hyperactive import RandomSearchOptimizer, ParticleSwarmOptimizer
#import hyperactive

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


sgd = optimizers.SGD(lr=0.01)
adam = optimizers.Adam(lr=0.01)

# this defines the structure of the model and print("time: {}".format(t2-t1))the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": [adam, sgd]},
    "keras.fit.0": {"epochs": [5], "batch_size": range(10, 101), "verbose": [2]},
    "keras.layers.Conv2D.1": {
        "filters": range(4, 101),
        "kernel_size": [3, 5, 7],
        "activation": ["sigmoid", "relu", "tanh"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.3": {
        "filters": range(4, 101),
        "kernel_size": [3, 5, 7],
        "activation": ["sigmoid", "relu", "tanh"],
    },
    "keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
    "keras.layers.Flatten.5": {},
    "keras.layers.Dense.6": {"units": range(4, 201), "activation": ["sigmoid", "relu", "tanh"]},
    "keras.layers.Dense.7": {"units": range(4, 201), "activation": ["sigmoid", "relu", "tanh"]},
    "keras.layers.Dense.8": {"units": [10], "activation": ["softmax"]},
}

Optimizer = ParticleSwarmOptimizer(search_config, n_iter=10, n_part=10, metric='accuracy', cv=0.8, w=0.7, c_k=2.0, c_s=2.0)

t1 = time.time()
Optimizer.fit(X_train, y_train)
t2 = time.time()

print("time: {}".format(t2-t1))

# predict from test data
Optimizer.predict(X_test)
score = Optimizer.score(X_test, y_test)

print("test score: {}".format(score))
