# Step 5 — XGBoost Classification Models

> Maps to **§ 4.2** of [`docs/Capstone_Paper.pdf`](../docs/Capstone_Paper.pdf).

The binary target is `sensitive` (LN_IC50 < 0) vs. `resistant` (LN_IC50 ≥ 0).
Four classifiers were trained on the same 2 × 2 design as Step 4 — **mutation
data in/out** × **PCA in/out** — with PCA branches dropped from the final
discussion because they degraded both accuracy and interpretability.

## 5.1 Common training setup

```
from xgboost import XGBClassifier

clf = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss",
    objective="binary:logistic",
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train_cls)
y_proba = clf.predict_proba(X_test)[:, 1]
```

ROC AUC, confusion matrices, and feature importance are reported on the held-out
test split.

## 5.2 Configuration 1 — expression only, no PCA (Figures 9–11) ⭐

| Metric         | Value  |
|----------------|--------|
| ROC AUC        | **0.9701** |
| Overall accuracy | **96 %** |
| Precision (sensitive class) | 0.90 |
| Recall (sensitive class)    | 0.70 |
| F1 (sensitive class)        | 0.79 |
| Precision (resistant class) | 0.96 |
| Recall (resistant class)    | 0.99 |

**Confusion matrix breakdown:**

| | Predicted resistant | Predicted sensitive |
|---|---|---|
| Actual resistant | 2,699 (TN) | 27 (FP) |
| Actual sensitive | 112 (FN)  | 257 (TP) |

The 112 false negatives reflect a class-imbalance penalty on the sensitive (minority)
class — the model is excellent at confirming resistance and conservative on
sensitivity calls.

**Top feature:** drug target **RNA polymerase**, by a large margin over the next
predictors. Other strong predictors include compound **ML323**, pathway feature
**Protein stability and degradation**, and pharmacological screening parameters
(`MIN_CONC`, `MAX_CONC`).

## 5.3 Configuration 2 — expression + mutation, no PCA (Figures 12–14) ⭐

| Metric         | Value  |
|----------------|--------|
| ROC AUC        | **0.9708** |
| Overall accuracy | **96 %** |
| Precision (sensitive class) | 0.91 |
| Recall (sensitive class)    | 0.70 |

**Confusion matrix breakdown:**

| | Predicted resistant | Predicted sensitive |
|---|---|---|
| Actual resistant | ≈ 2,700 (TN) | small (FP) |
| Actual sensitive | 111 (FN) | 258 (TP) |

**Notable feature-importance changes from Configuration 1:**

- A combined indicator covering **BCL-2 / BCL-XL / BFL-1 / MCL-1** (the BCL-2
  family of apoptosis-evasion proteins) enters the top predictors. LUAD cell
  lines that overexpress these survival proteins resist drugs that would
  otherwise trigger apoptosis (Wesarg et al., 2007), so this is a biologically
  coherent signal.
- Drug-specific features such as **Ribociclib** and **Bortezomib** retain
  importance.
- The HDAC family (`HDAC1/2/3/8`) remains a strong predictor block.

Adding mutation data does not materially move ROC AUC, but it **introduces
clinically interpretable features** the expression-only model could not surface.

## 5.4 Configurations 3 & 4 — with PCA (excluded from main results)

Both PCA variants were trained and evaluated. Versus Configurations 1 & 2:

- Accuracy and recall on the sensitive class **dropped**.
- The most discriminative genomic and drug-target features could no longer be
  retrieved from the linear combinations of components.

Since the goal of the classification framing is both to **predict** and to
**interpret** drug response, the PCA classification branches were excluded from
the final discussion in the paper.

## 5.5 Take-aways

- A well-tuned XGBoost classifier reaches **ROC AUC ≈ 0.97** on the held-out
  split with or without mutation data — the model has excellent discriminative
  capacity on LUAD pharmacogenomics.
- **Gene expression carries the dominant predictive signal.** Driver mutations
  contribute marginally to ROC AUC but substantially to biological
  interpretability (BCL-2 family, ASXL1).
- **PCA is the wrong dimensionality reduction for this classifier.** It degrades
  both accuracy and interpretability. If dimensionality reduction is needed,
  tree-based feature selection (e.g. importance-thresholded subsets) is the
  better path.

---

⬅️ Previous: [`04_regression_models.md`](04_regression_models.md)
🏠 Back to: [`../README.md`](../README.md)
