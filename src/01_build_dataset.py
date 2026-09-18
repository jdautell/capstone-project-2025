"""Merge the five GDSC sources into the LUAD modeling table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/new_merged_subset_v3.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    details = pd.read_csv(data_dir / "Cell_Lines_Details.csv").rename(
        columns={"COSMIC identifier": "COSMIC_ID"}
    )
    luad_samples = details.loc[details["Cancer Type"] == "LUAD"].copy()
    luad_ids = set(luad_samples["COSMIC_ID"].dropna().astype(int))

    dose_response = pd.read_csv(data_dir / "GDSC2_fitted_dose_response_27Oct23.csv")
    dose_response = dose_response.loc[dose_response["COSMIC_ID"].isin(luad_ids)]

    expression_path = data_dir / "Cell_line_RMA_proc_basalExp.csv"
    expression_columns = pd.read_csv(expression_path, nrows=0).columns.tolist()
    luad_columns = []
    for column in expression_columns[1:]:
        if not column.startswith("DATA."):
            continue
        try:
            cosmic_id = int(column.removeprefix("DATA.").split(".")[0])
        except ValueError:
            continue
        if cosmic_id in luad_ids:
            luad_columns.append(column)

    expression = pd.read_csv(
        expression_path,
        usecols=[expression_columns[0], *luad_columns],
    )
    gene_column = expression.columns[0]
    expression = expression.dropna(subset=[gene_column]).drop_duplicates(gene_column)
    expression = expression.rename(columns={gene_column: "GENE_SYMBOLS"})
    expression = expression.rename(
        columns={column: column.removeprefix("DATA.").split(".")[0] for column in luad_columns}
    )
    expression = expression.set_index("GENE_SYMBOLS").T
    expression.index = expression.index.astype(int)
    expression.index.name = "COSMIC_ID"
    expression = expression.reset_index()

    merged = dose_response.merge(luad_samples, on="COSMIC_ID", suffixes=("_drug", "_cell"))
    merged = merged.merge(expression, on="COSMIC_ID", how="inner")

    compounds = pd.read_csv(data_dir / "screened_compounds_rel_8.5.csv")
    merged = merged.merge(compounds, on="DRUG_ID", how="left", suffixes=("", "_comp"))

    mutations = pd.read_csv(data_dir / "mutations_all_20230202.csv")
    mutations = mutations.loc[mutations["cancer_driver"].eq(True)]
    mutations = mutations.dropna(subset=["model_id", "gene_symbol"])
    mutations = mutations.drop_duplicates(["model_id", "gene_symbol"])
    mutation_matrix = (
        mutations.assign(present=1)
        .pivot(index="model_id", columns="gene_symbol", values="present")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    merged["SANGER_MODEL_ID"] = merged["SANGER_MODEL_ID"].astype(str)
    mutation_matrix["model_id"] = mutation_matrix["model_id"].astype(str)
    merged = merged.merge(
        mutation_matrix,
        left_on="SANGER_MODEL_ID",
        right_on="model_id",
        how="left",
    ).drop(columns=["model_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved {merged.shape[0]:,} rows x {merged.shape[1]:,} columns to {args.output}")


if __name__ == "__main__":
    main()

