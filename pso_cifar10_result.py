import time
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras import optimizers

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

learning_time = []
test_score = []

model = Sequential()
model.add(Conv2D(filters=42, kernel_size=(5,5), padding='same', activation='tanh', input_shape=(32, 32, 3)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=51, kernel_size=(7,7), padding='same', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(95, activation='relu'))
model.add(Dense(101, activation='relu'))
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

