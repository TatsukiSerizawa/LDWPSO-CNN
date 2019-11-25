import time
import numpy as np
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras import optimizers

#(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#X_train = X_train.reshape(60000, 28, 28, 1)
#X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

learning_time = []
test_score = []

model = Sequential()
#model.add(Conv2D(filters=6, kernel_size=(5,5), padding='same', activation='sigmoid', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='same', activation='sigmoid', input_shape=(32, 32, 3)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(120, activation='sigmoid'))
model.add(Dense(84, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

sgd = optimizers.SGD(lr=0.01)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=10, 
                    batch_size=10,
                    verbose=1)

score = model.evaluate(X_test, y_test)
print("Accuracy: {}".format(score[1]))

