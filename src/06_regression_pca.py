"""XGBoost regression with 100-component PCA and randomized search."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import randint, uniform
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import (
    REGRESSION_PCA_CATEGORICAL,
    REGRESSION_PCA_EXCLUDE,
    build_feature_matrix,
    load_merged,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/new_merged_subset_v3.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/regression_pca"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_merged(args.data)
    target = frame["LN_IC50"]
    features = build_feature_matrix(frame, REGRESSION_PCA_CATEGORICAL, REGRESSION_PCA_EXCLUDE)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    pca = PCA(n_components=100, random_state=42)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    parameter_distributions = {
        "n_estimators": randint(50, 300),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    }
    search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42, tree_method="hist"),
        param_distributions=parameter_distributions,
        n_iter=20,
        cv=3,
        scoring="r2",
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(x_train_pca, y_train)
    predictions = search.best_estimator_.predict(x_test_pca)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.txt").write_text(
        f"Best parameters: {search.best_params_}\nR2: {r2:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}\n",
        encoding="utf-8",
    )
    print(f"R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")


if __name__ == "__main__":
    main()

