from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score, roc_curve, auc
)
import numpy as np


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    avg = "weighted" if len(np.unique(y_true)) > 2 else "binary"
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)

    roc_data = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        if y_proba.ndim == 2:
            proba = y_proba[:, 1]
        else:
            proba = y_proba
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    return metrics, cm, roc_data


def compute_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
    }
