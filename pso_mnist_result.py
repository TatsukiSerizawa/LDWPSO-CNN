import time
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras import optimizers

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

learning_time = []
test_score = []

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(7,7), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=100, kernel_size=(7,7), padding='same', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(66, activation='tanh'))
model.add(Dense(10, activation='softmax'))

sgd = optimizers.SGD(lr=0.01)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)


history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=5, 
                    batch_size=99,
                    verbose=1)

score = model.evaluate(X_test, y_test)
print("Accuracy: {}".format(score[1]))

