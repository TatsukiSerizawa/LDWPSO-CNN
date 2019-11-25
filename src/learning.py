import os
import pandas as pd
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random as rn
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping 


# Label binarization
def change_label(label_num):
    if label_num >= 5.0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    label = pd.read_csv('../../data/deap_dataset/participant_ratings_female.csv')

    # delete unnecessary labels
    label = label.drop(columns=["Trial", "Start_time", "Dominance", "Liking", "Familiarity"])

    # Label binarization
    label['Valence'] = label['Valence'].apply(change_label)
    label['Arousal'] = label['Arousal'].apply(change_label)

    # Labeling
    X = []
    Y = []
    for k in range(1, 6):
        for i in range(40):
            #files = os.listdir("./timeseries_plot_images/s" + str(k) + "/" + str(i))
            #男女を分けて行う（今回はfemale）
            files = os.listdir("../../data/timeseries_image/timeseries_plot_images3_female15_gray77/s" + str(k) + "/" + str(i))
            print(str(i) + " ", end="")
            for j in range(len(files)):
                #n = os.path.join("./timeseries_plot_images/s" + str(k) + "/" + str(i) + "/", files[j])
                #男女を分けて行う（今回はfemale）
                n = os.path.join("../../data/timeseries_image/timeseries_plot_images3_female15_gray77/s" + str(k) + "/" + str(i) + "/", files[j])
                image = cv2.imread(n)
                b,g,r = cv2.split(image)
                image = cv2.merge([r,g,b])
                X.append(image)
                Y.append(label.loc[40*(k-1)+i, "Arousal"])
        print("")
    
    # データ分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    #行列に変換
    X_train=np.array(X_train, dtype="float32")
    X_test=np.array(X_test, dtype="float32")

    # Normalization
    X_train /= 255.0
    X_test /= 255.0

    # クラス行列に変換
    y_train = to_categorical(Y_train)
    y_test = to_categorical(Y_test)


    # 学習におけるランダム値固定
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(7)
    rn.seed(7)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )

    tf.set_random_seed(7)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


    # CNN model
    model = Sequential()

    model.add(Conv2D(filters=100, kernel_size=(2,2), padding='Same',# kernel_regularizer=regularizers.l2(0.01),
                    activation='relu', input_shape=(77,77,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=4, kernel_size=(2,2), padding='Same',# kernel_regularizer=regularizers.l2(0.01),
                    activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(103, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer = "SGD",
                loss = "categorical_crossentropy",
                metrics=["accuracy"])
    model.summary()

    # patience: 何回連続で損失関数が向上しなければストップするか
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto') 

    # Learning
    history = model.fit(X_train, y_train, 
                        batch_size=10,
                        epochs=10,
                        verbose=1,
                        validation_data=(X_test, y_test)
                        #callbacks=[early_stopping]
                        )