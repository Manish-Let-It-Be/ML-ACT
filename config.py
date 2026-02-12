import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

BUILTIN_DATASETS = {
    "Iris (Classification)": "iris.csv",
    "Wine (Classification)": "wine.csv",
    "Breast Cancer (Classification)": "breast_cancer.csv",
    "Housing (Regression)": "housing.csv",
}

CLASSIFICATION_ALGORITHMS = [
    "Naive Bayes",
    "Decision Tree (ID3 - Entropy)",
    "CART",
    "KNN",
    "Logistic Regression",
    "Perceptron (Single Layer)",
    "Multi-Layer Perceptron",
    "SVM (Linear)",
    "SVM (Non-linear)",
]

REGRESSION_ALGORITHMS = [
    "Linear Regression",
    "Multiple Linear Regression",
]

SEMI_SUPERVISED_ALGORITHMS = [
    "Label Propagation",
    "Self-Training Classifier",
]

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
