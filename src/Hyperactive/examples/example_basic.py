from sklearn.datasets import load_iris

from hyperactive import RandomSearchOptimizer

iris_data = load_iris()
<<<<<<< HEAD
X, y = iris_data.data, iris_data.target
=======
X = iris_data.data
y = iris_data.target
>>>>>>> dbd0b511032907d8ce3be0ca13570fb6c3f0fa6e

search_config = {
    "sklearn.ensemble.RandomForestClassifier": {"n_estimators": range(10, 100, 10)}
}

<<<<<<< HEAD
opt = RandomSearchOptimizer(search_config, n_iter=10)
opt.fit(X, y)
=======
Optimizer = RandomSearchOptimizer(search_config, n_iter=10, verbosity=0)
Optimizer.fit(X, y)
>>>>>>> dbd0b511032907d8ce3be0ca13570fb6c3f0fa6e
