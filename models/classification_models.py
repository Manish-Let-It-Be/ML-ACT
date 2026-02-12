from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_classification_models(selected_algorithms, hyperparams=None):
    if hyperparams is None:
        hyperparams = {}

    models = {}

    if "Naive Bayes" in selected_algorithms:
        models["Naive Bayes"] = GaussianNB()

    if "Decision Tree (ID3 - Entropy)" in selected_algorithms:
        params = hyperparams.get("Decision Tree (ID3 - Entropy)", {})
        models["Decision Tree (ID3 - Entropy)"] = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
        )

    if "CART" in selected_algorithms:
        params = hyperparams.get("CART", {})
        models["CART"] = DecisionTreeClassifier(
            criterion="gini",
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
        )

    if "KNN" in selected_algorithms:
        params = hyperparams.get("KNN", {})
        models["KNN"] = KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
        )

    if "Logistic Regression" in selected_algorithms:
        params = hyperparams.get("Logistic Regression", {})
        models["Logistic Regression"] = LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            C=params.get("C", 1.0),
        )

    if "Perceptron (Single Layer)" in selected_algorithms:
        params = hyperparams.get("Perceptron (Single Layer)", {})
        models["Perceptron (Single Layer)"] = Perceptron(
            max_iter=params.get("max_iter", 1000),
        )

    if "Multi-Layer Perceptron" in selected_algorithms:
        params = hyperparams.get("Multi-Layer Perceptron", {})
        hidden = params.get("hidden_layer_sizes", (100,))
        models["Multi-Layer Perceptron"] = MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=params.get("max_iter", 1000),
        )

    if "SVM (Linear)" in selected_algorithms:
        params = hyperparams.get("SVM (Linear)", {})
        models["SVM (Linear)"] = SVC(
            kernel="linear",
            C=params.get("C", 1.0),
            probability=True,
        )

    if "SVM (Non-linear)" in selected_algorithms:
        params = hyperparams.get("SVM (Non-linear)", {})
        models["SVM (Non-linear)"] = SVC(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            probability=True,
        )

    return models
