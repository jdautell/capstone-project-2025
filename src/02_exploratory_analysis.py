"""Reproduce the mutation, drug-sensitivity, and correlation summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pointbiserialr

from common import load_merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/new_merged_subset_v3.csv"))
    parser.add_argument("--figures", type=Path, default=Path("outputs/eda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_merged(args.data)
    args.figures.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(frame["LN_IC50"], bins=50, color="skyblue")
    plt.title("Distribution of LN_IC50")
    plt.xlabel("LN_IC50")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.figures / "ln_ic50_distribution.png", dpi=200)
    plt.close()

    mutation_columns = [
        column
        for column in frame.columns
        if column.endswith("_y") and frame[column].nunique(dropna=True) == 2
    ]
    correlations = []
    for gene in mutation_columns:
        subset = frame[[gene, "LN_IC50"]].dropna()
        correlation, p_value = pointbiserialr(subset[gene], subset["LN_IC50"])
        correlations.append((gene, correlation, p_value))

    correlation_frame = pd.DataFrame(
        correlations,
        columns=["Gene", "Correlation", "P-value"],
    ).sort_values("Correlation", key=lambda values: values.abs(), ascending=False)
    correlation_frame.to_csv(args.figures / "mutation_lnic50_correlations.csv", index=False)

    top = correlation_frame.head(15)
    plt.figure(figsize=(10, 5))
    plt.bar(top["Gene"], top["Correlation"], color="coral")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Point-Biserial Correlation with LN_IC50")
    plt.title("Top Mutated Genes Correlated with Drug Sensitivity")
    plt.tight_layout()
    plt.savefig(args.figures / "mutation_lnic50_correlations.png", dpi=200)
    plt.close()

    drug_effectiveness = frame.groupby("DRUG_NAME")["LN_IC50"].mean().sort_values().head(5)
    drug_effectiveness.to_csv(args.figures / "top_effective_drugs.csv")
    drug_effectiveness.plot(kind="barh", figsize=(8, 6), color="teal")
    plt.xlabel("Average LN_IC50 (Lower = More Effective)")
    plt.title("Top 5 Most Effective Drugs in LUAD Cell Lines")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(args.figures / "top_effective_drugs.png", dpi=200)
    plt.close()

    print(f"Analyzed {frame['COSMIC_ID'].nunique()} cell lines and {frame['DRUG_NAME'].nunique()} drugs")


if __name__ == "__main__":
    main()

