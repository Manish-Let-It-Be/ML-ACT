import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve


def plot_metric_comparison(results, metric, task_type="classification"):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    values = [results[n].get("metrics", {}).get(metric, 0) for n in names]
    colors = sns.color_palette("viridis", len(names))
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} Comparison")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    has_data = False
    for name, data in results.items():
        roc = data.get("roc_data")
        if roc is not None:
            ax.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={roc['auc']:.3f})")
            has_data = True
    if has_data:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        return fig
    plt.close(fig)
    return None


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [feature_names[i] for i in indices]
    vals = importances[indices]
    ax.barh(names[::-1], vals[::-1], color=sns.color_palette("magma", len(names)))
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_learning_curve(model, X, y, title="Learning Curve", cv=5, scoring="accuracy"):
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring=scoring
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
        ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
        ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation Score")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        return None
