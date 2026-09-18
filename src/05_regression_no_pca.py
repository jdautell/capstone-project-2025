"""XGBoost regression without PCA, preserving feature-level importance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import (
    REGRESSION_NO_PCA_CATEGORICAL,
    REGRESSION_NO_PCA_EXCLUDE,
    build_feature_matrix,
    load_merged,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/new_merged_subset_v3.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/regression_no_pca"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_merged(args.data)
    target = frame["LN_IC50"]
    features = build_feature_matrix(
        frame,
        REGRESSION_NO_PCA_CATEGORICAL,
        REGRESSION_NO_PCA_EXCLUDE,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    model = XGBRegressor(
        colsample_bytree=0.6063865008880857,
        learning_rate=0.0792681476866447,
        max_depth=6,
        n_estimators=160,
        subsample=0.8733054075301833,
        random_state=42,
        tree_method="hist",
    )
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.txt").write_text(
        f"R2: {r2:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}\n",
        encoding="utf-8",
    )

    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_indices)), importances[sorted_indices])
    plt.xticks(range(len(sorted_indices)), features.columns[sorted_indices], rotation=90)
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Features in XGBoost Regression Model - No PCA")
    plt.tight_layout()
    plt.savefig(args.output / "feature_importance.png", dpi=200)
    plt.close()
    print(f"R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")


if __name__ == "__main__":
    main()

