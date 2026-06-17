# Step 2 — Preprocessing and Feature Engineering

> Maps to **§ 2.3** of [`docs/Capstone_Paper.pdf`](../docs/Capstone_Paper.pdf).

## 2.1 Missing-value handling

No feature exhibited > 50 % missing values, so no column was dropped on missingness
grounds. Any residual `NaN`s in the continuous gene-expression columns were imputed
to zero on the standardized scale (equivalent to "no deviation from the mean") for
downstream PCA stability.

## 2.2 Categorical encoding — one-hot

All categorical variables (drug name, drug target, target pathway, cell-line growth
characteristics) are converted to binary columns via `OneHotEncoder` so they are
compatible with `XGBRegressor` / `XGBClassifier`:

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cat_cols = ["DRUG_NAME", "PUTATIVE_TARGET", "TARGET_PATHWAY",
            "Growth Properties", "Cancer Type (matching TCGA label)"]
num_cols = [c for c in df.columns if c not in cat_cols + ["LN_IC50"]]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(),                                            num_cols),
])
```

One-hot expansion adds several hundred binary columns, dominated by the drug-name
and drug-target categories.

## 2.3 Continuous standardization — `StandardScaler`

All continuous features (gene expression + drug-screening metadata like
`MIN_CONC` / `MAX_CONC`) are z-scored:

> `x' = (x − μ) / σ` so each feature has μ = 0, σ = 1.

XGBoost does not strictly require scaling, but **PCA does**, and the standardized
form makes the two pipeline branches (PCA-on / PCA-off) comparable on the same
feature representation.

## 2.4 Memory optimization — `float32` cast

The wide multi-omic table has ~17,000 numerical features. To control memory and
accelerate training, all floating-point columns are downcast to `float32`:

```
num_block = num_block.astype("float32")
```

XGBoost handles `float32` natively and there is no measurable accuracy loss
relative to `float64` on this dataset.

## 2.5 Target binarization (for classification)

Following Menden et al. (2013), the continuous LN_IC50 target is binarized at
zero:

| LN_IC50 range | Class label   | Interpretation                                  |
|---------------|---------------|------------------------------------------------|
| `< 0`         | **sensitive** | Drug inhibits at sub-1 μM equivalents (potent)  |
| `≥ 0`         | **resistant** | Drug requires ≥ 1 μM equivalents to inhibit     |

```
y_cls = (df["LN_IC50"] < 0).astype(int)
```

This binarization is conservative (only the most-sensitive responses populate
class 1), trading recall on the sensitive class for higher confidence in the
positive predictions — and matches the convention in pharmacogenomic ML.

## 2.6 Optional PCA branch — 100 components

For the PCA pipeline branch, dimensionality is reduced to **100 components** after
standardization:

```
from sklearn.decomposition import PCA
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)
```

100 components were selected to capture the dominant variance while keeping the
training matrix tractable on a single machine. PCA is applied only in the
configurations that explicitly test the speed/interpretability trade-off in
Step 4 / Step 5.

---

⬅️ Previous: [`01_data_ingestion.md`](01_data_ingestion.md)
➡️ Next: [`03_exploratory_data_analysis.md`](03_exploratory_data_analysis.md)
