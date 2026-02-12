from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier
from sklearn.svm import SVC
import numpy as np


def get_semi_supervised_models(selected_algorithms, hyperparams=None):
    if hyperparams is None:
        hyperparams = {}

    models = {}

    if "Label Propagation" in selected_algorithms:
        models["Label Propagation"] = LabelPropagation(max_iter=1000)

    if "Self-Training Classifier" in selected_algorithms:
        base = SVC(kernel="rbf", probability=True, gamma="scale")
        models["Self-Training Classifier"] = SelfTrainingClassifier(base_estimator=base)

    return models


def prepare_semi_supervised_data(X_train, y_train, unlabeled_fraction=0.3):
    n = len(y_train)
    n_unlabeled = int(n * unlabeled_fraction)
    rng = np.random.RandomState(42)
    unlabeled_idx = rng.choice(n, size=n_unlabeled, replace=False)
    y_semi = y_train.copy()
    y_semi[unlabeled_idx] = -1
    return y_semi
