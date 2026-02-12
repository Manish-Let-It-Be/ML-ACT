import streamlit as st
import pandas as pd
import numpy as np
import os

from config import (
    BUILTIN_DATASETS, DATASETS_DIR, MODELS_DIR, REPORTS_DIR,
    CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS, SEMI_SUPERVISED_ALGORITHMS,
)
from preprocessing.missing_handler import handle_missing_values
from preprocessing.normalization import normalize_data
from preprocessing.outlier_detection import remove_outliers_zscore
from models.regression_models import get_regression_models
from models.classification_models import get_classification_models
from models.semi_supervised import get_semi_supervised_models, prepare_semi_supervised_data
from evaluation.metrics import compute_classification_metrics, compute_regression_metrics
from evaluation.comparison import create_comparison_table, get_best_model
from visualization.plots import (
    plot_metric_comparison, plot_confusion_matrix, plot_roc_curves,
    plot_feature_importance, plot_learning_curve,
)
from tuning.hyperparameter_tuning import perform_grid_search, perform_cross_validation, PARAM_GRIDS
from kaggle_integration.kaggle_loader import download_kaggle_dataset, detect_target_column

from ollama_integration.report_generator import generate_ai_report, check_ollama_status, list_ollama_models
from utils.helpers import save_model, load_dataset, get_dataset_info, format_dataset_info
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="ML Comparison Framework",
    page_icon="֎",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>

:root {
    --bg: #f8f9fb;
    --card-bg: #ffffff;
    --text-primary: #111827;
    --text-secondary: #6b7280;
    --accent: #2563eb;
    --border: #e5e7eb;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0e1117;
        --card-bg: #262730;
        --text-primary: #fafafa;
        --text-secondary: #c9cccf;
        --accent: #3b82f6;
        --border: #3d4043;
    }
}

/* Streamlit dark mode detection */
.stApp[data-theme="dark"] {
    --bg: #0e1117;
    --card-bg: #262730;
    --text-primary: #fafafa;
    --text-secondary: #c9cccf;
    --accent: #3b82f6;
    --border: #3d4043;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
            

            
.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: var(--text-secondary) !important;
    max-width: 700px;
    margin: 0 auto 2rem auto;
}

.section-title {
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--text-primary) !important;
    margin-bottom: 2rem;
}

.divider {
    height: 1px;
    background: var(--border);
    margin: 4rem 0;
}

.feature-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 2rem;
    transition: all 0.2s ease;
    height: 100%;
}

.feature-card:hover {
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
    transform: translateY(-3px);
}

.feature-title {
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    color: var(--text-primary) !important;
}

.feature-text {
    color: var(--text-secondary) !important;
    font-size: 0.95rem;
}

.step {
    display: flex;
    align-items: flex-start;
    gap: 15px;
    margin-bottom: 1.5rem;
}

.step-circle {
    min-width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #eef2ff;
    color: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.9rem;
}

.primary-btn > button {
    background-color: var(--accent);
    background-color: #dc2626;    font-weight: 500;
}
            
.footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f8f9fa;
        padding: 20px;
        text-align: center;
        border-top: 1px solid #ddd;
        z-index: 999;
    }
.footer-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
    }

</style>
""", unsafe_allow_html=True)


if "results" not in st.session_state:
    st.session_state.results = {}
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []
if "show_landing" not in st.session_state:
    st.session_state.show_landing = True


def render_sidebar():
    with st.sidebar:
        st.markdown("## Dataset")

        data_source = st.radio("Source", ["Built-in", "Kaggle"], horizontal=True)

        df = None
        dataset_name = ""

        if data_source == "Built-in":
            dataset_choice = st.selectbox("Select Dataset", list(BUILTIN_DATASETS.keys()))
            dataset_name = dataset_choice
            filepath = os.path.join(DATASETS_DIR, BUILTIN_DATASETS[dataset_choice])
            df = load_dataset(filepath)
        else:
            st.markdown("**Popular Datasets:**")
            examples = [
                "vikrishnan/iris-dataset",
                "uciml/iris",
                "mlg-ulb/creditcardfraud",
                "uciml/breast-cancer-wisconsin-data",
                "uciml/pima-indians-diabetes-database"
            ]
            selected_example = st.selectbox("Quick Select", ["Custom..."] + examples)
            
            if selected_example == "Custom...":
                kaggle_name = st.text_input("Kaggle Dataset", placeholder="owner/dataset-name")
            else:
                kaggle_name = st.text_input("Kaggle Dataset", value=selected_example)
            
            if st.button("Download Dataset") and kaggle_name:
                with st.spinner("Downloading from Kaggle..."):
                    result_df, fname, error = download_kaggle_dataset(kaggle_name)
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.session_state.kaggle_df = result_df
                        st.session_state.kaggle_name = fname
                        st.success(f"Downloaded: {fname}")

            if "kaggle_df" in st.session_state:
                df = st.session_state.kaggle_df
                dataset_name = st.session_state.get("kaggle_name", "Kaggle Dataset")

        if df is not None:
            st.markdown("---")
            default_target = detect_target_column(df) if data_source == "Kaggle" else df.columns[-1]
            default_idx = df.columns.tolist().index(default_target) if default_target in df.columns else len(df.columns) - 1
            target_col = st.selectbox("Target Column", df.columns.tolist(), index=default_idx)

            st.markdown("---")
            st.markdown("## Task Type")
            task_type = st.radio("Select Task", ["Classification", "Regression"], horizontal=True)

            st.markdown("---")
            st.markdown("## Preprocessing")

            missing_method = st.selectbox(
                "Missing Values",
                ["None", "Drop Rows", "Mean Imputation", "Median Imputation"],
            )

            normalization = st.selectbox(
                "Normalization",
                ["None", "StandardScaler", "MinMaxScaler"],
            )

            remove_outliers = st.checkbox("Remove Outliers (Z-Score)", value=False)
            outlier_threshold = 3.0
            if remove_outliers:
                outlier_threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.5)

            test_size = st.slider("Test Split Ratio", 0.1, 0.5, 0.2, 0.05)

            st.markdown("---")
            st.markdown("## Algorithms")

            if task_type == "Classification":
                available = CLASSIFICATION_ALGORITHMS + SEMI_SUPERVISED_ALGORITHMS
            else:
                available = REGRESSION_ALGORITHMS

            selected_algorithms = st.multiselect("Select Algorithms", available, default=available[:3])

            st.markdown("---")
            st.markdown("## Advanced Options")

            enable_cv = st.checkbox("Cross Validation (K-Fold)", value=False)
            cv_folds = 5
            if enable_cv:
                cv_folds = st.slider("Number of Folds", 2, 10, 5)

            enable_grid_search = st.checkbox("GridSearchCV", value=False)

            st.markdown("---")
            hyperparams = {}
            with st.expander("Hyperparameter Settings"):
                for algo in selected_algorithms:
                    if algo == "KNN":
                        st.markdown(f"**{algo}**")
                        k = st.slider("n_neighbors", 1, 21, 5, 2, key=f"knn_k")
                        hyperparams["KNN"] = {"n_neighbors": k}
                    elif algo in ["Decision Tree (ID3 - Entropy)", "CART"]:
                        st.markdown(f"**{algo}**")
                        md = st.slider("max_depth", 1, 30, 5, key=f"{algo}_md")
                        mss = st.slider("min_samples_split", 2, 20, 2, key=f"{algo}_mss")
                        hyperparams[algo] = {"max_depth": md, "min_samples_split": mss}
                    elif algo in ["Logistic Regression", "SVM (Linear)"]:
                        st.markdown(f"**{algo}**")
                        c = st.select_slider("C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0, key=f"{algo}_c")
                        hyperparams[algo] = {"C": c}
                    elif algo == "SVM (Non-linear)":
                        st.markdown(f"**{algo}**")
                        c = st.select_slider("C", [0.01, 0.1, 1.0, 10.0], value=1.0, key="svm_nl_c")
                        kernel = st.selectbox("Kernel", ["rbf", "poly", "sigmoid"], key="svm_nl_k")
                        hyperparams[algo] = {"C": c, "kernel": kernel}
                    elif algo == "Multi-Layer Perceptron":
                        st.markdown(f"**{algo}**")
                        layers = st.text_input("Hidden Layers (comma-sep)", "100", key="mlp_layers")
                        try:
                            hidden = tuple(int(x.strip()) for x in layers.split(","))
                        except ValueError:
                            hidden = (100,)
                        max_iter = st.slider("max_iter", 100, 2000, 1000, 100, key="mlp_iter")
                        hyperparams[algo] = {"hidden_layer_sizes": hidden, "max_iter": max_iter}

            return {
                "df": df,
                "dataset_name": dataset_name,
                "target_col": target_col,
                "task_type": task_type,
                "missing_method": missing_method,
                "normalization": normalization,
                "remove_outliers": remove_outliers,
                "outlier_threshold": outlier_threshold,
                "test_size": test_size,
                "selected_algorithms": selected_algorithms,
                "enable_cv": enable_cv,
                "cv_folds": cv_folds,
                "enable_grid_search": enable_grid_search,
                "hyperparams": hyperparams,
            }
    return None


def preprocess_data(df, target_col, config):
    df_processed = df.copy()

    missing_map = {
        "Drop Rows": "drop",
        "Mean Imputation": "mean",
        "Median Imputation": "median",
    }
    if config["missing_method"] != "None":
        df_processed = handle_missing_values(df_processed, missing_map[config["missing_method"]])

    if config["remove_outliers"]:
        df_processed = remove_outliers_zscore(df_processed, config["outlier_threshold"])

    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

    norm_map = {"StandardScaler": "standard", "MinMaxScaler": "minmax"}
    if config["normalization"] != "None":
        X_values, _ = normalize_data(X.values, norm_map[config["normalization"]])
        X = pd.DataFrame(X_values, columns=X.columns)

    return X, y


def train_models(X, y, config):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=42
    )

    task_type = config["task_type"]
    selected = config["selected_algorithms"]
    hyperparams = config["hyperparams"]

    if task_type == "Classification":
        clf_algos = [a for a in selected if a in CLASSIFICATION_ALGORITHMS]
        semi_algos = [a for a in selected if a in SEMI_SUPERVISED_ALGORITHMS]

        models = get_classification_models(clf_algos, hyperparams)
        semi_models = get_semi_supervised_models(semi_algos, hyperparams)
        models.update(semi_models)
    else:
        models = get_regression_models(selected)

    total = len(models)
    progress_bar = st.progress(0, text="Training models...")
    status_text = st.empty()

    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training: {name} ({i+1}/{total})")

        try:
            is_semi = name in SEMI_SUPERVISED_ALGORITHMS

            if config["enable_grid_search"] and name in PARAM_GRIDS and not is_semi:
                scoring = "accuracy" if task_type == "Classification" else "r2"
                model, best_params, best_score = perform_grid_search(
                    model, X_train, y_train, PARAM_GRIDS[name], cv=config["cv_folds"], scoring=scoring
                )

            if is_semi:
                y_semi = prepare_semi_supervised_data(X_train.values if hasattr(X_train, 'values') else X_train, y_train.values if hasattr(y_train, 'values') else y_train)
                model.fit(X_train, y_semi)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if task_type == "Classification":
                y_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception:
                        pass
                metrics, cm, roc_data = compute_classification_metrics(y_test, y_pred, y_proba)
                results[name] = {
                    "metrics": metrics,
                    "confusion_matrix": cm,
                    "roc_data": roc_data,
                    "model": model,
                }
            else:
                metrics = compute_regression_metrics(y_test, y_pred)
                results[name] = {"metrics": metrics, "model": model}

            if config["enable_cv"] and not is_semi:
                scoring = "accuracy" if task_type == "Classification" else "r2"
                cv_mean, cv_std = perform_cross_validation(model, X, y, cv=config["cv_folds"], scoring=scoring)
                results[name]["cv_mean"] = cv_mean
                results[name]["cv_std"] = cv_std

            model_path = save_model(model, name)
            results[name]["model_path"] = model_path

        except Exception as e:
            results[name] = {"metrics": {}, "error": str(e)}

        progress_bar.progress((i + 1) / total, text=f"Completed: {name}")

    status_text.text("All models trained!")
    progress_bar.empty()

    return results, X_train, X_test, y_train, y_test


def render_results(results, config, X_train, X_test, y_train, y_test, feature_names):
    task_type = config["task_type"]

    comparison_df = create_comparison_table(results)
    if config["enable_cv"]:
        cv_data = []
        for name, data in results.items():
            if "cv_mean" in data:
                cv_data.append({"Algorithm": name, "CV Mean": data["cv_mean"], "CV Std": data["cv_std"]})
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            comparison_df = comparison_df.merge(cv_df, on="Algorithm", how="left")

    tab_metrics, tab_charts, tab_details = st.tabs(["Metrics Table", "Visualizations", "Model Details"])

    with tab_metrics:
        st.markdown("### Performance Comparison")
        st.dataframe(comparison_df.style.format(precision=4), width='stretch')

        if task_type == "Classification":
            best_name, best_score = get_best_model(results, "Accuracy")
        else:
            best_name, best_score = get_best_model(results, "R2 Score")

        if best_name:
            st.success(f"**Best Model: {best_name}** (Score: {best_score:.4f})")

        st.markdown("### Model Leaderboard")
        if task_type == "Classification":
            metric_key = "Accuracy"
        else:
            metric_key = "R2 Score"

        leaderboard = []
        for name, data in results.items():
            score = data.get("metrics", {}).get(metric_key, 0)
            leaderboard.append({"Rank": 0, "Algorithm": name, metric_key: score})
        leaderboard = sorted(leaderboard, key=lambda x: x[metric_key], reverse=True)
        for i, row in enumerate(leaderboard):
            row["Rank"] = i + 1
        st.dataframe(pd.DataFrame(leaderboard), width='stretch')

    with tab_charts:
        if task_type == "Classification":
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_metric_comparison(results, "Accuracy", task_type)
                st.pyplot(fig)
            with col2:
                fig = plot_metric_comparison(results, "F1-Score", task_type)
                st.pyplot(fig)

            st.markdown("### Confusion Matrices")
            cols = st.columns(min(3, len(results)))
            for idx, (name, data) in enumerate(results.items()):
                if "confusion_matrix" in data:
                    with cols[idx % len(cols)]:
                        fig = plot_confusion_matrix(data["confusion_matrix"], title=name)
                        st.pyplot(fig)

            roc_fig = plot_roc_curves(results)
            if roc_fig:
                st.markdown("### ROC Curves")
                st.pyplot(roc_fig)
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_metric_comparison(results, "R2 Score", task_type)
                st.pyplot(fig)
            with col2:
                fig = plot_metric_comparison(results, "RMSE", task_type)
                st.pyplot(fig)

        st.markdown("### Feature Importance")
        for name, data in results.items():
            model = data.get("model")
            if model and hasattr(model, "feature_importances_"):
                fig = plot_feature_importance(model, feature_names, title=f"{name} - Feature Importance")
                if fig:
                    st.pyplot(fig)

        st.markdown("### Learning Curves")
        for name, data in results.items():
            model = data.get("model")
            if model and not isinstance(model, type):
                with st.expander(f"Learning Curve: {name}"):
                    try:
                        X_combined = np.vstack([X_train, X_test])
                        y_combined = np.concatenate([y_train, y_test])
                        lc_scoring = "accuracy" if task_type == "Classification" else "r2"
                        from sklearn.base import clone
                        model_clone = clone(model)
                        fig = plot_learning_curve(model_clone, X_combined, y_combined, title=name, cv=3, scoring=lc_scoring)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.info("Learning curve not available for this model.")
                    except Exception as e:
                        st.info(f"Learning curve not available: {str(e)[:80]}")

    with tab_details:
        for name, data in results.items():
            with st.expander(f"{name}"):
                if "error" in data:
                    st.error(f"Error: {data['error']}")
                else:
                    st.json(data.get("metrics", {}))
                    if "model_path" in data:
                        st.caption(f"Saved to: {data['model_path']}")


def render_ai_report(results, config, dataset_info_str):
    st.markdown("### AI-Powered Analysis")

    ollama_available = check_ollama_status()

    analysis_type = st.radio("Analysis Type", ["AI-Powered (Ollama)", "Automated (No AI)"], horizontal=True)

    if analysis_type == "AI-Powered (Ollama)":
        if ollama_available:
            models = list_ollama_models()
            selected_model = st.selectbox("Ollama Model", models if models else ["mistral"], key="ollama_model")

            if st.button("Generate AI Report", type="primary"):
                comparison_df = create_comparison_table(results)
                metrics_str = comparison_df.to_string(index=False)
                algorithms = list(results.keys())

                with st.spinner("Generating AI analysis..."):
                    report, error = generate_ai_report(
                        dataset_info_str, algorithms, metrics_str,
                        model=selected_model,
                    )

                if error:
                    st.error(f"Error: {error}")
                else:
                    st.markdown(report)
                    report_path = os.path.join(REPORTS_DIR, "ai_report.txt")
                    with open(report_path, "w") as f:
                        f.write(report)
                    st.download_button("Download Report", report, file_name="ai_report.txt", mime="text/plain")
        else:
            st.warning(
                "Ollama is not running locally. To use AI analysis:\n\n"
                "1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`\n"
                "2. Start it: `ollama serve`\n"
                "3. Pull a model: `ollama pull mistral`"
            )
    else:
        if results:
            task_type = config["task_type"]
            metric_key = "Accuracy" if task_type == "Classification" else "R2 Score"
            best_name, best_score = get_best_model(results, metric_key)

            analysis_lines = [f"**Best Model: {best_name}** with {metric_key} = {best_score:.4f}\n"]
            analysis_lines.append("**Performance Summary:**\n")
            for name, data in results.items():
                metrics = data.get("metrics", {})
                scores = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                analysis_lines.append(f"- **{name}**: {scores}")

            if config.get("enable_cv"):
                analysis_lines.append("\n**Cross-Validation Results:**\n")
                for name, data in results.items():
                    if "cv_mean" in data:
                        diff = abs(list(data["metrics"].values())[0] - data["cv_mean"])
                        flag = " (possible overfitting)" if diff > 0.05 else ""
                        analysis_lines.append(f"- **{name}**: CV Mean = {data['cv_mean']:.4f} +/- {data['cv_std']:.4f}{flag}")

            st.markdown("\n".join(analysis_lines))

            report_text = "\n".join(analysis_lines)
            st.download_button("Download Analysis", report_text, file_name="analysis_report.txt", mime="text/plain")


def render_educational_section():
    st.markdown("---")
    st.markdown("## ML Theory & Education")

    tab_pac, tab_bias, tab_version, tab_error = st.tabs(["PAC Learning", "Bias-Variance", "Version Space", "Error Bounds"])

    with tab_pac:
        st.markdown("""
### PAC Learning (Probably Approximately Correct)

PAC learning is a framework for mathematical analysis of machine learning. It was proposed by Leslie Valiant in 1984.

**Core Idea:** A concept class is PAC-learnable if there exists an algorithm that, for any distribution over the input space and any target concept in the class, can produce a hypothesis that is approximately correct with high probability.

**Key Parameters:**
- **Epsilon (error):** The maximum acceptable error rate. The learner must find a hypothesis with error at most epsilon.
- **Delta (confidence):** The probability of failure. The learner must succeed with probability at least (1 - delta).

**Sample Complexity:** The number of training examples needed:

$$m \\geq \\frac{1}{\\epsilon}\\left(\\ln|H| + \\ln\\frac{1}{\\delta}\\right)$$

Where |H| is the size of the hypothesis space.

**Practical Implications:**
- More complex hypothesis spaces require more training data
- Higher accuracy requirements need more samples
- Higher confidence requirements need more samples
- PAC learning provides theoretical guarantees for generalization
        """)

    with tab_bias:
        st.markdown("""
### Bias-Variance Tradeoff

The bias-variance tradeoff is fundamental to understanding model performance.

**Bias:** Error from erroneous assumptions in the learning algorithm. High bias leads to underfitting.
- Example: Linear regression on non-linear data

**Variance:** Error from sensitivity to fluctuations in the training set. High variance leads to overfitting.
- Example: Deep decision tree memorizing noise

**Total Error = Bias^2 + Variance + Irreducible Error**

| Model Type | Bias | Variance | Typical Behavior |
|-----------|------|----------|-----------------|
| Linear Regression | High | Low | Underfitting |
| Decision Tree (deep) | Low | High | Overfitting |
| KNN (small k) | Low | High | Overfitting |
| KNN (large k) | High | Low | Underfitting |
| SVM (RBF, high C) | Low | High | Overfitting |
| Naive Bayes | High | Low | Underfitting |

**How to Manage:**
- **High Bias:** Use more complex models, add features, reduce regularization
- **High Variance:** Get more data, use regularization, use simpler models, use ensemble methods
        """)

    with tab_version:
        st.markdown("""
### Version Space

The version space is the set of all hypotheses consistent with the training data.

**Concept:**
- Given a hypothesis space H and training examples D
- Version Space VS(H,D) = {h in H | h is consistent with D}
- As more examples are seen, the version space shrinks

**Boundaries:**
- **S (Specific Boundary):** Most specific hypotheses consistent with positive examples
- **G (General Boundary):** Most general hypotheses consistent with negative examples
- All valid hypotheses lie between S and G

**Candidate Elimination Algorithm:**
1. Initialize S to the most specific hypothesis
2. Initialize G to the most general hypothesis
3. For each positive example: generalize S
4. For each negative example: specialize G
5. Remove inconsistent hypotheses

**Convergence:** The version space converges to a single hypothesis when enough examples are provided, assuming the target concept is in H.
        """)

    with tab_error:
        st.markdown("""
### Error Bounds in Machine Learning

**Training Error vs Generalization Error:**
- Training error: performance on training data
- Generalization error: expected performance on unseen data
- Goal: minimize generalization error

**Hoeffding's Inequality:**

$$P[|E_{in} - E_{out}| > \\epsilon] \\leq 2|H|e^{-2\\epsilon^2 N}$$

Where:
- E_in = training error
- E_out = generalization error
- |H| = hypothesis space size
- N = number of training examples

**VC Dimension (Vapnik-Chervonenkis):**
- Measures the capacity of a model
- Largest set of points the model can shatter
- Higher VC dimension = more complex model = needs more data

**VC Generalization Bound:**

$$E_{out} \\leq E_{in} + \\sqrt{\\frac{8}{N}\\ln\\frac{4m_H(2N)}{\\delta}}$$

**Practical Guidelines:**
- Rule of thumb: need ~10x VC dimension training examples
- Cross-validation provides empirical error estimates
- Regularization controls model complexity to prevent overfitting
        """)

def render_landing_page():

    # HERO SECTION
    st.markdown("""
    <div style="text-align:center;">
        <div class="hero-title">ML Algorithms Comparison Tool</div>
        <div class="hero-subtitle">
            Train, compare and analyze machine learning models in one unified interface.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # FEATURES SECTION
    st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Multiple Datasets</div>
            <div class="feature-text">
                Use built-in datasets or import directly from Kaggle with automatic target detection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">10+ Algorithms</div>
            <div class="feature-text">
                Compare classification, regression and semi-supervised learning models side-by-side.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Visual Analytics</div>
            <div class="feature-text">
                Evaluate performance using ROC curves, confusion matrices, learning curves and more.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Hyperparameter Tuning</div>
            <div class="feature-text">
                Improve results with GridSearchCV and cross-validation strategies.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # HOW IT WORKS SECTION
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="step">
        <div class="step-circle">1</div>
        <div>Select a dataset from built-in options or Kaggle.</div>
    </div>
    <div class="step">
        <div class="step-circle">2</div>
        <div>Choose your task type and configure preprocessing.</div>
    </div>
    <div class="step">
        <div class="step-circle">3</div>
        <div>Select algorithms and tuning options.</div>
    </div>
    <div class="step">
        <div class="step-circle">4</div>
        <div>Train models and compare performance metrics.</div>
    </div>
    <div class="step">
        <div class="step-circle">5</div>
        <div>Analyze results using visual reports and AI insights.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col = st.columns([1,2,1])
    with col[1]:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Get Started", use_container_width=True):
            st.session_state.show_landing = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)



def render_prediction_interface(sidebar_config):
    if not st.session_state.results:
        return
    
    st.markdown("---")
    st.markdown("### 🔮 Make Custom Predictions")
    
    df = sidebar_config["df"]
    target_col = sidebar_config["target_col"]
    feature_cols = [col for col in df.columns if col != target_col]
    
    st.markdown("Enter custom values to get predictions from trained models:")
    
    input_data = {}
    cols = st.columns(min(3, len(feature_cols)))
    
    for idx, col in enumerate(feature_cols):
        with cols[idx % len(cols)]:
            if df[col].dtype in [np.int64, np.float64]:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val, key=f"pred_{col}")
            else:
                unique_vals = df[col].unique().tolist()
                input_data[col] = st.selectbox(f"{col}", unique_vals, key=f"pred_{col}")
    
    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([input_data])
        
        # Apply same preprocessing
        non_numeric = input_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            input_df = pd.get_dummies(input_df, columns=non_numeric, drop_first=True)
        
        # Align columns with training data
        X_train = st.session_state.get("X_train")
        if X_train is not None:
            for col in X_train.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X_train.columns]
        
        st.markdown("#### Predictions:")
        predictions = {}
        for name, data in st.session_state.results.items():
            if "model" in data and "error" not in data:
                try:
                    pred = data["model"].predict(input_df)[0]
                    predictions[name] = pred
                except Exception as e:
                    predictions[name] = f"Error: {str(e)[:50]}"
        
        pred_df = pd.DataFrame(list(predictions.items()), columns=["Model", "Prediction"])
        st.dataframe(pred_df, width='stretch')


def render_footer():
    st.markdown("""
        <div class="footer-container">
            <p style="color: #888; margin-bottom: 15px;">Made with ❤️ for ML Enthusiasts</p>
            <a href="https://github.com/Manish-Let-It-Be" target="_blank" style="text-decoration: none;">
                <div style="display: inline-flex; align-items: center; background: #24292e; color: white; padding: 10px 20px; border-radius: 2rem; font-weight: 600; transition: background 0.3s;">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="filter: invert(1); margin-right: 10px;">
                    GitHub
                </div>
            </a>
            <p style="color: #aaa; font-size: 0.8rem; margin-top: 15px;">&copy; 2026 ML ACT</p>
        </div>
    """, unsafe_allow_html=True)


def main():
    if st.session_state.show_landing:
        render_landing_page()
        render_footer()
        return
    
    st.markdown('<p class="main-header">ML ACT</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train, compare, and analyze machine learning algorithms with an interactive interface</p>', unsafe_allow_html=True)

    sidebar_config = render_sidebar()

    if sidebar_config is None:
        st.info("Select a dataset from the sidebar to get started.")
        render_educational_section()
        render_footer()
        return

    df = sidebar_config["df"]
    target_col = sidebar_config["target_col"]

    main_tabs = st.tabs(["Data Explorer", "Train & Compare", "AI Analysis", "ML Theory"])

    with main_tabs[0]:
        st.markdown("### Dataset Preview")
        info = get_dataset_info(df, sidebar_config["dataset_name"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", info["shape"][0])
        col2.metric("Columns", info["shape"][1])
        col3.metric("Numeric Cols", info["numeric_columns"])
        col4.metric("Missing Values", info["missing_values"])

        st.dataframe(df.head(50), width='stretch')

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Statistical Summary")
            st.dataframe(df.describe(), width='stretch')
        with col_b:
            st.markdown("#### Target Distribution")
            target_counts = df[target_col].value_counts()
            if len(target_counts) <= 20:
                st.bar_chart(target_counts)
            else:
                st.line_chart(df[target_col].sort_values().reset_index(drop=True))

    with main_tabs[1]:
        st.markdown("### Train & Compare Models")

        if not sidebar_config["selected_algorithms"]:
            st.warning("Please select at least one algorithm from the sidebar.")
        else:
            st.markdown(f"**Task:** {sidebar_config['task_type']} | "
                        f"**Algorithms:** {len(sidebar_config['selected_algorithms'])} | "
                        f"**Test Split:** {sidebar_config['test_size']}")

            if st.button("Train All Models", type="primary", width='stretch'):
                X, y = preprocess_data(df, target_col, sidebar_config)
                feature_names = list(X.columns) if hasattr(X, 'columns') else [f"Feature_{i}" for i in range(X.shape[1])]

                results, X_train, X_test, y_train, y_test = train_models(X, y, sidebar_config)

                st.session_state.results = results
                st.session_state.feature_names = feature_names
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.dataset_info_str = format_dataset_info(info)

                history_entry = {
                    "dataset": sidebar_config["dataset_name"],
                    "task": sidebar_config["task_type"],
                    "algorithms": list(results.keys()),
                    "metrics": {n: d.get("metrics", {}) for n, d in results.items()},
                }
                st.session_state.history.append(history_entry)

            if st.session_state.results:
                render_results(
                    st.session_state.results, sidebar_config,
                    st.session_state.get("X_train"),
                    st.session_state.get("X_test"),
                    st.session_state.get("y_train"),
                    st.session_state.get("y_test"),
                    st.session_state.get("feature_names", []),
                )
                
                render_prediction_interface(sidebar_config)

    with main_tabs[2]:
        if st.session_state.results:
            render_ai_report(
                st.session_state.results, sidebar_config,
                st.session_state.get("dataset_info_str", ""),
            )
        else:
            st.info("Train models first to generate AI analysis.")

    with main_tabs[3]:
        render_educational_section()

    if st.session_state.history:
        with st.expander("Experiment History"):
            for i, entry in enumerate(st.session_state.history):
                st.markdown(f"**Run {i+1}:** {entry['dataset']} ({entry['task']}) - {len(entry['algorithms'])} algorithms")
    
    render_footer()


if __name__ == "__main__":
    main()
