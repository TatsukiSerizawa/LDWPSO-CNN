# coding: utf-8

# Basic sklearn example

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from hyperactive import SimulatedAnnealingOptimizer
from hyperactive import ParticleSwarmOptimizer
import time

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": [3, 4, 5, 6],
        "criterion": ["gini", "entropy"],
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(2, 21),
    }
}

# start point
start_point = {
    "sklearn.ensemble.RandomForestClassifier.0": {
        "n_estimators": [30],
        "max_depth": [6],
        "criterion": ["entropy"],
        "min_samples_split": [12],
        "min_samples_leaf": [16],
    }
}

#Optimizer = SimulatedAnnealingOptimizer(search_config, n_iter=100, n_jobs=4)
Optimizer = ParticleSwarmOptimizer(search_config, 
                                    n_iter=10,  # number of iterations to perform
                                    metric="accuracy", 
                                    n_jobs=1, 
                                    cv=3, 
                                    verbosity=1, 
                                    random_state=None, 
                                    warm_start=start_point,  # Hyperparameter configuration to start from
                                    memory=True,  # Stores explored evaluations in a dictionary to save computing time
                                    scatter_init=False,  # Chooses better initial position by training on multiple random positions with smaller training dataset 
                                    n_part=10,  # number of particles
                                    w=0.5,  # interia factor
                                    c_k=0.5,  # cognitive factor
                                    c_s=0.9)  # social factor

# search best hyperparameter for given data
t1 = time.time()
Optimizer.fit(X_train, y_train)
t2 = time.time()
print("time: {}".format(t2-t1))

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)

print("test accracy: {}".format(score))

# Optimizer.export('hoge.txt')