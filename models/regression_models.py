from sklearn.linear_model import LinearRegression


def get_regression_models(selected_algorithms):
    models = {}
    if "Linear Regression" in selected_algorithms:
        models["Linear Regression"] = LinearRegression()
    if "Multiple Linear Regression" in selected_algorithms:
        models["Multiple Linear Regression"] = LinearRegression()
    return models
