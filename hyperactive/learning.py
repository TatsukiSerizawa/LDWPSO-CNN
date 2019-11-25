import os
import pandas as pd
import numpy as np
import cv2
from keras.utils import to_categorical
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
                Y.append(df.loc[40*(k-1)+i, "Arousal"])
        print("")