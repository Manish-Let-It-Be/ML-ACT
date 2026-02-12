from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np


PARAM_GRIDS = {
    "KNN": {"n_neighbors": [3, 5, 7, 9, 11]},
    "Decision Tree (ID3 - Entropy)": {"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10]},
    "CART": {"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10]},
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "SVM (Linear)": {"C": [0.01, 0.1, 1, 10]},
    "SVM (Non-linear)": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "Multi-Layer Perceptron": {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "max_iter": [500, 1000]},
}


def perform_grid_search(model, X_train, y_train, param_grid, cv=5, scoring="accuracy"):
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def perform_cross_validation(model, X, y, cv=5, scoring="accuracy"):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std()
