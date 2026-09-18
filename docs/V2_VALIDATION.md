# V2: leakage-aware validation

## Why a V2 was needed

The original ROC AUC of approximately 0.97 came from a random response-level
80/20 split. A row represents a cell-line/drug pair, so the same cell line and the
same drug can occur in both partitions. The score is useful as an exploratory
in-distribution result, but it does not establish performance on a new cell line or
a new compound.

## What changed

`src/08_v2_grouped_validation.py` evaluates classification and regression under
three repeated holdout schemes:

1. **Random response pairs** — retained only to compare V2 with the original work.
2. **Unseen cell lines** — no `COSMIC_ID` occurs in both training and test data.
3. **Unseen drugs** — no `DRUG_ID` occurs in both training and test data.

The V2 also fits encoding only on training data; excludes response-derived values,
curve IDs, and row identifiers; removes direct drug identity in the unseen-drug
test; audits group overlap; and reports uncertainty over repeated holdouts.

## Run

```bash
python src/08_v2_grouped_validation.py \
  --data data/processed/new_merged_subset_v3.csv \
  --output outputs/v2_validation \
  --repeats 5
```

Verify the complete pipeline without the original GDSC files:

```bash
python src/08_v2_grouped_validation.py --smoke-test --repeats 2 --n-estimators 20
```

The primary claims should use the **unseen-cell** and **unseen-drug** rows in
`summary_metrics.csv`, including their standard deviations. A lower grouped score
is expected: it answers a harder and more honest question. These cell-line results
are preclinical and do not imply patient-level clinical performance.
