"""Leakage-aware V2 validation for LUAD drug-sensitivity prediction.

The original target definitions are retained for comparison, while evaluation is
changed: preprocessing is trained only on training data, response-derived columns
and identifiers are excluded, and random-pair, unseen-cell and unseen-drug splits
are reported separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


TARGET = "LN_IC50"
CELL_GROUP = "COSMIC_ID"
DRUG_GROUP = "DRUG_ID"

# Row IDs and values derived from the same response experiment are not predictors.
ALWAYS_EXCLUDE = {
    TARGET,
    "AUC",
    "RMSE",
    "Z_SCORE",
    "NLME_RESULT_ID",
    "NLME_CURVE_ID",
    "COSMIC_ID",
    "DRUG_ID",
    "SANGER_MODEL_ID",
    "CELL_LINE_NAME",
    "Sample Name",
    "model_id",
    "DATASET",
    "COMPANY_ID",
    "MIN_CONC",
    "MAX_CONC",
    "WEBRELEASE",
}

# Direct compound identity cannot generalize to a compound absent from training.
DRUG_IDENTITY_COLUMNS = {"DRUG_ID", "DRUG_NAME", "DRUG_NAME_comp", "SYNONYMS"}


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train_index: np.ndarray
    test_index: np.ndarray
    group_column: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/new_merged_subset_v3.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/v2_validation"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run end-to-end on generated data; verifies code, not performance.",
    )
    return parser.parse_args()


def make_smoke_data(seed: int = 42) -> pd.DataFrame:
    """Create a small response-pair table with realistic repeated groups."""
    rng = np.random.default_rng(seed)
    cells = np.arange(1001, 1031)
    drugs = np.arange(201, 213)
    rows: list[dict[str, object]] = []
    cell_effect = dict(zip(cells, rng.normal(0, 0.8, len(cells))))
    drug_effect = dict(zip(drugs, rng.normal(0, 1.0, len(drugs))))
    for cell in cells:
        expression = rng.normal(size=12)
        for drug in rng.choice(drugs, size=8, replace=False):
            ln_ic50 = (
                0.8 * expression[0]
                - 0.6 * expression[1]
                + cell_effect[cell]
                + drug_effect[drug]
                + rng.normal(0, 0.45)
            )
            row: dict[str, object] = {
                CELL_GROUP: cell,
                DRUG_GROUP: drug,
                "SANGER_MODEL_ID": f"SIDM{cell}",
                "CELL_LINE_NAME": f"LUAD_{cell}",
                "DRUG_NAME": f"Compound_{drug}",
                "PUTATIVE_TARGET": f"Target_{drug % 4}",
                "PATHWAY_NAME": f"Pathway_{drug % 3}",
                "AUC": 1 / (1 + np.exp(-ln_ic50)),
                "RMSE": abs(rng.normal(0.1, 0.03)),
                TARGET: ln_ic50,
            }
            row.update({f"GENE_{i:02d}": value for i, value in enumerate(expression)})
            row.update({f"MUT_{i:02d}": int(rng.random() < 0.12) for i in range(5)})
            rows.append(row)
    return pd.DataFrame(rows)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.smoke_test:
        frame = make_smoke_data(args.random_state)
    else:
        frame = pd.read_csv(args.data, low_memory=False)
    required = {TARGET, CELL_GROUP, DRUG_GROUP}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(required)).reset_index(drop=True)
    frame["sensitive"] = frame[TARGET].lt(0).astype(np.int8)
    return frame


def build_splits(frame: pd.DataFrame, args: argparse.Namespace) -> list[SplitSpec]:
    splits: list[SplitSpec] = []
    stratified = StratifiedShuffleSplit(
        n_splits=args.repeats,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    for fold, (train, test) in enumerate(stratified.split(frame, frame["sensitive"]), 1):
        splits.append(SplitSpec(f"random_pair_{fold}", train, test, None))

    for label, group_column in (("unseen_cell", CELL_GROUP), ("unseen_drug", DRUG_GROUP)):
        grouped = GroupShuffleSplit(
            n_splits=args.repeats,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        for fold, (train, test) in enumerate(
            grouped.split(frame, groups=frame[group_column]), 1
        ):
            splits.append(SplitSpec(f"{label}_{fold}", train, test, group_column))
    return splits


def feature_columns(frame: pd.DataFrame, unseen_drug: bool) -> tuple[list[str], list[str]]:
    excluded = set(ALWAYS_EXCLUDE) | {"sensitive"}
    if unseen_drug:
        excluded |= DRUG_IDENTITY_COLUMNS
    candidates = [column for column in frame.columns if column not in excluded]
    categorical = [
        column
        for column in candidates
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    numeric = [column for column in candidates if column not in categorical]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    # XGBoost handles numeric NaNs; scaling is unnecessary for decision trees.
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        sparse_threshold=1.0,
    )


def safe_auc(y_true: pd.Series, scores: np.ndarray, metric: str) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    if metric == "roc":
        return float(roc_auc_score(y_true, scores))
    return float(average_precision_score(y_true, scores))


def evaluate_split(
    frame: pd.DataFrame, split: SplitSpec, args: argparse.Namespace
) -> dict[str, object]:
    train = frame.iloc[split.train_index]
    test = frame.iloc[split.test_index]
    numeric, categorical = feature_columns(
        frame, unseen_drug=split.group_column == DRUG_GROUP
    )
    preprocessor = make_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(train[numeric + categorical])
    x_test = preprocessor.transform(test[numeric + categorical])
    if not sparse.issparse(x_train):
        x_train = sparse.csr_matrix(np.asarray(x_train, dtype=np.float32))
        x_test = sparse.csr_matrix(np.asarray(x_test, dtype=np.float32))

    shared = dict(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=args.random_state,
    )
    classifier = XGBClassifier(eval_metric="logloss", **shared)
    classifier.fit(x_train, train["sensitive"])
    probability = classifier.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(np.int8)

    regressor = XGBRegressor(objective="reg:squarederror", **shared)
    regressor.fit(x_train, train[TARGET])
    regression_prediction = regressor.predict(x_test)

    return {
        "split": split.name.rsplit("_", 1)[0],
        "fold": int(split.name.rsplit("_", 1)[1]),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_rate": float(train["sensitive"].mean()),
        "test_positive_rate": float(test["sensitive"].mean()),
        "cell_group_overlap": len(
            set(train[CELL_GROUP]).intersection(test[CELL_GROUP])
        ),
        "drug_group_overlap": len(
            set(train[DRUG_GROUP]).intersection(test[DRUG_GROUP])
        ),
        "n_numeric_features": len(numeric),
        "n_categorical_features": len(categorical),
        "roc_auc": safe_auc(test["sensitive"], probability, "roc"),
        "pr_auc": safe_auc(test["sensitive"], probability, "pr"),
        "accuracy": float(accuracy_score(test["sensitive"], prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test["sensitive"], prediction)
        ),
        "f1": float(f1_score(test["sensitive"], prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(test["sensitive"], prediction)),
        "r2": float(r2_score(test[TARGET], regression_prediction)),
        "rmse": float(mean_squared_error(test[TARGET], regression_prediction) ** 0.5),
        "mae": float(mean_absolute_error(test[TARGET], regression_prediction)),
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "mcc",
        "r2",
        "rmse",
        "mae",
    ]
    summary = results.groupby("split")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    frame = load_data(args)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in build_splits(frame, args):
        print(f"Evaluating {split.name} ...", flush=True)
        rows.append(evaluate_split(frame, split, args))

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(args.output / "fold_metrics.csv", index=False)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)
    metadata = {
        "source": "generated smoke-test data" if args.smoke_test else str(args.data),
        "rows": len(frame),
        "unique_cell_lines": int(frame[CELL_GROUP].nunique()),
        "unique_drugs": int(frame[DRUG_GROUP].nunique()),
        "target_rule": "LN_IC50 < 0",
        "primary_estimates": ["unseen_cell", "unseen_drug"],
        "note": "Random-pair results do not prove unseen-group generalization.",
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nV2 summary (mean across holdouts):")
    print(summary.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
