# Step 4 — XGBoost Regression Models

> Maps to **§ 4.1** of [`docs/Capstone_Paper.pdf`](../docs/Capstone_Paper.pdf).

Four XGBoost regressors were trained on a 2 × 2 design — **mutation data in/out**
× **PCA in/out** — plus a Lasso baseline.

## 4.1 Common training setup

```
from xgboost import XGBRegressor

reg = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

All models share an identical 80/20 train/test split stratified by cell line and
the seed `random_state=42`. Metrics reported below are on the held-out test set.

## 4.2 Configuration 1 — expression only, no PCA (Figure 7)

**Purpose:** maximum interpretability. Used to surface the most-important raw
features without PCA mixing.

**Top features identified:** drug target **RNA polymerase**, drug
**Sepantronium bromide (YM155)**, drug-target **antioxidant proteins**, drug-target
**metabolism**, drug-target **mitosis pathway**, gene-level features **ASXL1**,
**DLGAP3**, **SAYSD1**, and the compound **Romidepsin**.

> Quantitative R²/RMSE/MAE are not reported for this configuration in the paper;
> the configuration is used purely for feature-importance interpretation.

## 4.3 Configuration 2 — expression + mutation, no PCA (Figure 8)

**Purpose:** test whether driver mutations contribute predictive signal once
expression is already in the model.

**Key observation:** the only mutation feature to enter the top-20 importance
ranking is **ASXL1** (chromatin modifier; Jafarbeik-Iravani et al., 2024). Expression-derived features still dominate.

The bulk of the top-20 list is unchanged from Configuration 1: RNA polymerase target,
drug-pathway descriptors, antioxidant response elements.

## 4.4 Configuration 3 — expression only, PCA-100

**Purpose:** quantify the runtime / performance trade-off of dimensionality
reduction on a 17 k-feature space.

| Metric | Value  |
|--------|--------|
| R²     | 0.7943 |
| RMSE   | 1.2418 |
| MAE    | 0.9345 |

PCA preserves the majority of the predictive signal at much lower computational
cost, at the cost of losing feature-level biological interpretability (the
components are linear mixtures of the original features).

## 4.5 Configuration 4 — expression + mutation, PCA-100 ⭐ **best regression**

| Metric | Value  |
|--------|--------|
| R²     | **0.7976** |
| RMSE   | **1.2316** |
| MAE    | **0.9321** |

Adding mutation data to the PCA pipeline yields the best regression performance,
marginally above Config 3. Interpretation is constrained by PCA, so this
configuration is most appropriate for **deployment / ranking** rather than
biological discovery.

## 4.6 Lasso baseline

A `Lasso` regressor was fit as a linear baseline to confirm XGBoost is the right
choice on this feature space.

| Metric | Value  |
|--------|--------|
| R²     | 0.6771 |
| RMSE   | 1.5558 |
| MAE    | 1.2373 |

Lasso's strong feature-sparsity assumption underperforms in the presence of
17 k correlated features and substantial non-linear interactions — confirming
the choice of gradient-boosted trees.

## 4.7 Summary

| Config         | R²     | RMSE   | MAE    | Notes                                                  |
|----------------|--------|--------|--------|--------------------------------------------------------|
| 1 (expr)       | —      | —      | —      | Used for feature importance only                       |
| 2 (expr + mut) | —      | —      | —      | ASXL1 mutation enters top-20                           |
| 3 (PCA-100)    | 0.7943 | 1.2418 | 0.9345 | Good speed / accuracy balance                          |
| 4 (PCA + mut)  | **0.7976** | **1.2316** | **0.9321** | **Best regression configuration**           |
| Lasso baseline | 0.6771 | 1.5558 | 1.2373 | Confirms non-linear interactions; not used downstream  |

---

⬅️ Previous: [`03_exploratory_data_analysis.md`](03_exploratory_data_analysis.md)
➡️ Next: [`05_classification_models.md`](05_classification_models.md)
