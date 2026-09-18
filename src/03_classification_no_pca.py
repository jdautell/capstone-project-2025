"""XGBoost sensitive/resistant classifier using the original row-level split."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import CLASSIFICATION_CATEGORICAL, CLASSIFICATION_EXCLUDE, build_feature_matrix, load_merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/new_merged_subset_v3.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/classification_no_pca"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_merged(args.data)
    target = frame["LN_IC50"].lt(0).astype(int)
    features = build_feature_matrix(frame, CLASSIFICATION_CATEGORICAL, CLASSIFICATION_EXCLUDE)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = XGBClassifier(
        n_estimators=160,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.87,
        colsample_bytree=0.61,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)
    probabilities = model.predict_proba(x_test_scaled)[:, 1]

    args.output.mkdir(parents=True, exist_ok=True)
    auc = roc_auc_score(y_test, probabilities)
    report = classification_report(y_test, predictions)
    (args.output / "metrics.txt").write_text(f"ROC AUC: {auc:.6f}\n\n{report}", encoding="utf-8")

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost Classifier")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output / "roc_curve.png", dpi=200)
    plt.close()

    matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(args.output / "confusion_matrix.png", dpi=200)
    plt.close()

    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_indices)), importances[sorted_indices])
    plt.xticks(range(len(sorted_indices)), features.columns[sorted_indices], rotation=90)
    plt.xlabel("Feature Importance")
    plt.ylabel("Importance Score")
    plt.title("Top 20 Features in XGBoost Classifier with Mutation Data")
    plt.tight_layout()
    plt.savefig(args.output / "feature_importance.png", dpi=200)
    plt.close()
    print(f"ROC AUC: {auc:.4f}; outputs saved to {args.output}")


if __name__ == "__main__":
    main()

