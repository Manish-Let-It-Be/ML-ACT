import pandas as pd


def create_comparison_table(results):
    rows = []
    for name, data in results.items():
        row = {"Algorithm": name}
        row.update(data.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def get_best_model(results, metric="Accuracy"):
    best_name = None
    best_score = -float("inf")
    for name, data in results.items():
        score = data.get("metrics", {}).get(metric, -float("inf"))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score
