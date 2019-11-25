import os
import pandas as pd
import numpy as np
import time
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

from hyperactive import RandomSearchOptimizer, ParticleSwarmOptimizer


# Label binarization
def change_label(label_num):
    if label_num >= 5.0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    label = pd.read_csv('../../data/deap_dataset/participant_ratings_male.csv')

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
            files = os.listdir("../../data/timeseries_image/timeseries_plot_images3_male15_gray77/s" + str(k) + "/" + str(i))
            print(str(i) + " ", end="")
            for j in range(len(files)):
                #n = os.path.join("./timeseries_plot_images/s" + str(k) + "/" + str(i) + "/", files[j])
                #男女を分けて行う（今回はfemale）
                n = os.path.join("../../data/timeseries_image/timeseries_plot_images3_male15_gray77/s" + str(k) + "/" + str(i) + "/", files[j])
                image = cv2.imread(n)
                b,g,r = cv2.split(image)
                image = cv2.merge([r,g,b])
                X.append(image)
                Y.append(label.loc[40*(k-1)+i, "Valence"])
        print("")
    
    # データ分割は最適化の中で行われてるので必要なし
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    #X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    #行列に変換
    X_train=np.array(X, dtype="float32")
   # X_test=np.array(X_test, dtype="float32")

    # Normalization
    X_train /= 255.0
    #X_test /= 255.0

    # クラス行列に変換
    y_train = to_categorical(Y)
    #y_test = to_categorical(Y_test)


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
    search_config = {
        "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["adam", "SGD"]},
        "keras.fit.0": {"epochs": [25], "batch_size": range(10, 51), "verbose": [1]},
        "keras.layers.Conv2D.1": {
            "filters": range(4, 101),
            "kernel_size": [2, 3, 4, 5, 6],
            "activation": ["relu"],
            "input_shape": [(77, 77, 3)],
        },
        "keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
        "keras.layers.Conv2D.3": {
            "filters": range(4, 101),
            "kernel_size": [2, 3, 4, 5, 6],
            "activation": ["relu"],
        },
        "keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
        "keras.layers.Flatten.5": {},
        #"keras.layers.Dense.6": {"units": range(512, 4097, 512), "activation": ["relu"]},
        "keras.layers.Dense.6": {"units": range(4, 1025), "activation": ["relu"]},
        "keras.layers.Dropout.7": {"rate": np.arange(0.1, 1.0, 0.1)},
        "keras.layers.Dense.8": {"units": [2], "activation": ["softmax"]},
    }

    start = time.time()
    #Optimizer = RandomSearchOptimizer(search_config, n_iter=1, cv=5, metric="accuracy")
    Optimizer = ParticleSwarmOptimizer(search_config, n_iter=30, cv=0.8, metric="accuracy", n_part=10, w=0.9, c_k=2.0, c_s=2.0)
    Optimizer.fit(X_train, y_train)
    end = time.time()

    print("time: {}".format(end-start) + "[sec]")