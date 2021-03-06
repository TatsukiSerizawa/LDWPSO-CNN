# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [1], "batch_size": [300], "verbose": [0]},
    "keras.layers.Dense.1": {"units": [1], "activation": ["softmax"]},
}


def test_keras():
    from hyperactive import HillClimbingOptimizer

    opt = HillClimbingOptimizer(search_config, 1)
    opt.fit(X, y)
    opt.predict(X)
    opt.score(X, y)


def test_keras_scores():
    from hyperactive import RandomSearchOptimizer

    ml_scores = [
        "accuracy",
        "binary_accuracy",
        "categorical_accuracy",
        "sparse_categorical_accuracy",
        "top_k_categorical_accuracy",
        "sparse_top_k_categorical_accuracy",
    ]

    for score in ml_scores:
        opt = RandomSearchOptimizer(search_config, 1, metric=score)
        assert opt._config_.metric == score
        opt.fit(X, y)
        assert opt._config_.metric == score
        opt.predict(X)
        assert opt._config_.metric == score
        opt.score(X, y)
        assert opt._config_.metric == score


def test_keras_losses():
    from hyperactive import RandomSearchOptimizer

    ml_losses = [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_logarithmic_error",
        "squared_hinge",
        "hinge",
        # "categorical_hinge",
        "logcosh",
        "categorical_crossentropy",
        # "sparse_categorical_crossentropy",
        "binary_crossentropy",
        "kullback_leibler_divergence",
        "poisson",
        "cosine_proximity",
    ]

    for loss in ml_losses:
        opt = RandomSearchOptimizer(search_config, 1, metric=loss)
        assert opt._config_.metric == loss
        opt.fit(X, y)
        assert opt._config_.metric == loss
        opt.predict(X)
        assert opt._config_.metric == loss
        opt.score(X, y)
        assert opt._config_.metric == loss


def test_keras_n_jobs():
    from hyperactive import HillClimbingOptimizer

    n_jobs_list = [1, 2]
    for n_jobs in n_jobs_list:
        opt = HillClimbingOptimizer(search_config, 1, n_jobs=n_jobs)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_keras_n_iter():
    from hyperactive import HillClimbingOptimizer

    n_iter_list = [0, 1, 2]
    for n_iter in n_iter_list:
        opt = HillClimbingOptimizer(search_config, n_iter)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_keras_cv():
    from hyperactive import HillClimbingOptimizer

    cv_list = [0.1, 0.5, 0.9, 2]
    for cv in cv_list:
        opt = HillClimbingOptimizer(search_config, 1, cv=cv)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_keras_verbosity():
    from hyperactive import HillClimbingOptimizer

    verbosity_list = [0, 1]
    for verbosity in verbosity_list:
        opt = HillClimbingOptimizer(search_config, 1, verbosity=verbosity)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_keras_random_state():
    from hyperactive import HillClimbingOptimizer

    random_state_list = [None, 0, 1]
    for random_state in random_state_list:
        opt = HillClimbingOptimizer(search_config, 1, random_state=random_state)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_keras_warm_start():
    from hyperactive import HillClimbingOptimizer

    warm_start = {
        "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
        "keras.fit.0": {"epochs": [1], "batch_size": [300], "verbose": [0]},
        "keras.layers.Dense.1": {"units": [1], "activation": ["softmax"]},
    }

    warm_start_list = [None, warm_start]
    for warm_start in warm_start_list:
        opt = HillClimbingOptimizer(search_config, 1, warm_start=warm_start)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_keras_memory():
    from hyperactive import HillClimbingOptimizer

    memory_list = [False, True]
    for memory in memory_list:
        opt = HillClimbingOptimizer(search_config, 1, memory=memory)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_keras_scatter_init():
    from hyperactive import HillClimbingOptimizer

    scatter_init_list = [False, 2]
    for scatter_init in scatter_init_list:
        opt = HillClimbingOptimizer(search_config, 1, scatter_init=scatter_init)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)
