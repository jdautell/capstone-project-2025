"""Shared preprocessing helpers for the capstone modeling scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MIXED_TYPE_COLUMNS = [
    "DRUG_NAME",
    "PUTATIVE_TARGET",
    "DRUG_NAME_comp",
    "SYNONYMS",
    "TARGET",
]

CLASSIFICATION_CATEGORICAL = [
    "TCGA_DESC",
    "Screen Medium",
    "Growth Properties",
    "GDSC\nTissue descriptor 1",
    "GDSC\nTissue\ndescriptor 2",
    "Cancer Type",
    "Microsatellite \ninstability Status (MSI)",
    "DRUG_NAME",
    "PUTATIVE_TARGET",
    "PATHWAY_NAME",
    "WEBRELEASE",
    "Sample Name",
    "Whole Exome Sequencing (WES)",
    "Copy Number Alterations (CNA)",
    "Gene Expression",
    "Methylation",
    "Drug\nResponse",
    "DRUG_NAME_comp",
    "SYNONYMS",
    "TARGET",
    "TARGET_PATHWAY",
    "SCREENING_SITE",
]

CLASSIFICATION_EXCLUDE = [
    "DATASET",
    "NLME_RESULT_ID",
    "NLME_CURVE_ID",
    "COSMIC_ID",
    "CELL_LINE_NAME",
    "SANGER_MODEL_ID",
    "AUC",
    "RMSE",
    "Z_SCORE",
    "LN_IC50",
]

REGRESSION_PCA_CATEGORICAL = CLASSIFICATION_CATEGORICAL
REGRESSION_PCA_EXCLUDE = CLASSIFICATION_EXCLUDE

REGRESSION_NO_PCA_CATEGORICAL = [
    "DATASET",
    "CELL_LINE_NAME",
    "SANGER_MODEL_ID",
    "TCGA_DESC",
    "DRUG_NAME",
    "PUTATIVE_TARGET",
    "PATHWAY_NAME",
    "WEBRELEASE",
    "Sample Name",
    "Whole Exome Sequencing (WES)",
    "Copy Number Alterations (CNA)",
    "Gene Expression",
    "Methylation",
    "Drug\nResponse",
    "GDSC\nTissue descriptor 1",
    "GDSC\nTissue\ndescriptor 2",
    "Cancer Type",
    "Microsatellite \ninstability Status (MSI)",
    "Screen Medium",
    "Growth Properties",
    "DRUG_NAME_comp",
    "SYNONYMS",
    "TARGET",
    "TARGET_PATHWAY",
    "SCREENING_SITE",
]

REGRESSION_NO_PCA_EXCLUDE = [
    "NLME_RESULT_ID",
    "NLME_CURVE_ID",
    "COSMIC_ID",
    "DRUG_ID",
    "COMPANY_ID",
    "MIN_CONC",
    "MAX_CONC",
    "AUC",
    "RMSE",
    "Z_SCORE",
    "LN_IC50",
]


def load_merged(path: str | Path) -> pd.DataFrame:
    """Load the processed cell-line/drug table and remove missing targets."""
    frame = pd.read_csv(Path(path), low_memory=False)
    return frame.dropna(subset=["LN_IC50"]).reset_index(drop=True)


def build_feature_matrix(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    excluded_columns: list[str],
) -> pd.DataFrame:
    """Reproduce the original numeric filtering and one-hot encoding."""
    data = frame.copy()
    categorical = [column for column in categorical_columns if column in data.columns]

    for column in MIXED_TYPE_COLUMNS:
        if column in data.columns:
            data[column] = data[column].astype(str)

    numeric = [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in excluded_columns
    ]
    missing_rate = data[numeric].isna().mean()
    valid_numeric = missing_rate[missing_rate < 0.5].index.tolist()

    data[categorical] = data[categorical].fillna("Unknown")
    categorical_frame = pd.get_dummies(data[categorical], drop_first=True)
    numeric_frame = data[valid_numeric].fillna(0).astype(np.float32)
    return pd.concat([numeric_frame, categorical_frame], axis=1).astype(np.float32)

