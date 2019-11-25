import time
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPool2D, Dropout
from keras.models import Model
from keras.models import Sequential

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), 
                activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), 
                activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='sgd',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

t1 = time.time()
history = model.fit(X_train, y_train, 
                    epochs=5, 
                    batch_size=500, 
                    verbose=1,
                    validation_data=(X_test, y_test))
t2 = time.time()
print("time: {}".format(t2-t1))

score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))