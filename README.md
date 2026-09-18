# Predicting Drug Sensitivity in Lung Adenocarcinoma Using Machine Learning

> XGBoost regression and classification on multi-omic GDSC data (gene expression +
> somatic driver mutations) to predict drug sensitivity in **Lung Adenocarcinoma**
> (LUAD) cell lines. The original exploratory random-pair split reached ROC AUC
> **0.97** and R² **0.80**; the V2 adds stricter unseen-cell and unseen-drug tests.

**Capstone — M.S. Biological Data Science · Arizona State University**
*LSC 585 Capstone II in Biological Data Science · Dr. Ken Sweat · May 2, 2025*
**Authors:** Jordan Dautelle · Hang Tran · David Yoon · Alli Warren

📄 **Full capstone paper:** [`docs/Capstone_Paper.pdf`](docs/Capstone_Paper.pdf) (28 pages, all figures + references)

---

## TL;DR results

| Task / Setup                              | Mutation data | PCA | R²     | RMSE   | MAE    | ROC AUC | Accuracy |
|-------------------------------------------|---------------|-----|--------|--------|--------|---------|----------|
| XGBoost regression (expression only)      | No            | No  | —      | —      | —      | —       | —        |
| XGBoost regression (+ mutation)           | Yes           | No  | —      | —      | —      | —       | —        |
| XGBoost regression (PCA-100)              | No            | Yes | 0.7943 | 1.2418 | 0.9345 | —       | —        |
| **XGBoost regression (PCA-100 + mut.)**   | **Yes**       | Yes | **0.7976** | **1.2316** | **0.9321** | — | — |
| **XGBoost classifier (expression only)**  | No            | No  | —      | —      | —      | **0.9701** | **96%** |
| **XGBoost classifier (+ mutation)**       | **Yes**       | No  | —      | —      | —      | **0.9708** | **96%** |
| Lasso regression baseline                 | No            | No  | 0.6771 | 1.5558 | 1.2373 | —       | —        |

(Cells marked "—" are used only for feature-importance interpretation; quantitative
metrics are reported only where they meaningfully compare against another configuration.)

---

## Project goal

Predict drug sensitivity in **Lung Adenocarcinoma (LUAD)** cell lines from multi-omic
features (gene expression + driver-mutation status + drug/pathway metadata) drawn from
the **Genomics of Drug Sensitivity in Cancer (GDSC)** project, using two
complementary framings:

1. **Regression** — predict the natural-log half-maximal inhibitory concentration
   (LN_IC50) directly. LN_IC50 is the standard pharmacological measure of drug
   sensitivity; lower means more sensitive.
2. **Classification** — binarize LN_IC50 at zero (`< 0` → **sensitive**;
   `≥ 0` → **resistant**) and learn a discriminator.

A secondary objective was to test whether **somatic mutation data adds predictive
signal over gene expression alone**, and whether the cost in interpretability of
**dimensionality reduction (PCA)** is justified.

---

## Dataset sources (publicly available — GDSC bulk download)

| File                                          | Role in the pipeline                                                  |
|-----------------------------------------------|-----------------------------------------------------------------------|
| `GDSC2_fitted_dose_response_27Oct23`          | Target variable — LN_IC50 per (cell line, drug)                       |
| `Cell_Lines_Details`                          | Metadata used to filter for LUAD-tissue samples                       |
| `Cell_line_RMA_proc_basalExp`                 | RMA-normalized gene-expression matrix (transposed to cell-line × gene)|
| `mutations_all_20230202`                      | Binary somatic mutation matrix on cancer driver genes                 |
| `screened_compounds_rel_8.5`                  | Drug-level metadata (name, target, pathway category)                  |

Bulk download portal: https://www.cancerrxgene.org/downloads/bulk_download

> The raw GDSC files are **not** committed to this repo (licensed, large). The
> step-by-step ingestion is documented in
> [`methodology/01_data_ingestion.md`](methodology/01_data_ingestion.md).

---

## Pipeline

```
   GDSC bulk download (5 files)
        │
        ▼  Filter Cell_Lines_Details by tissue == LUAD
        ▼  Transpose Cell_line_RMA_proc_basalExp (COSMIC ID rows)
   LUAD-restricted multi-omic table
        │
        ▼  Merge expression × dose-response × screened_compounds (COSMIC ID + DRUG_ID)
        ▼  Merge mutations via Sanger model_id → cell line
   Wide multi-omic dataset (~17,000 numerical features)
        │
        ▼  One-hot encode categoricals · StandardScaler · float32 cast
        ▼  Binarize LN_IC50 (< 0 → sensitive · ≥ 0 → resistant)
        ▼  PCA(100) optional branch
        │
        ▼  XGBoost regression (LN_IC50)  ┐
        ▼  XGBoost classifier (binary)   ┤  4 configurations each:
                                          │  × mutation in/out × PCA in/out
                                          ▼
   Feature importance, ROC AUC, confusion matrices, R²/RMSE/MAE
```

---

## Repository layout

```
capstone-project-2025/
├── README.md                                this file
├── LICENSE                                  MIT
├── .gitignore
├── requirements.txt                         Python dependencies (pinned)
├── src/                                     reproducible analysis code
│   ├── 01_build_dataset.py                  merge the five GDSC sources
│   ├── 02_exploratory_analysis.py           EDA and biological summaries
│   ├── 03_classification_no_pca.py          XGBoost classifier, raw features
│   ├── 04_classification_pca.py             XGBoost classifier, PCA-100
│   ├── 05_regression_no_pca.py              XGBoost regression, raw features
│   ├── 06_regression_pca.py                 XGBoost regression, PCA-100
│   ├── 07_classification_tuning.py           randomized classifier tuning
│   └── common.py                            shared preprocessing definitions
├── docs/
│   └── Capstone_Paper.pdf                   full deliverable paper (28 pages, all figures + references)
├── methodology/                             step-by-step methodology breakdown
│   ├── 01_data_ingestion.md
│   ├── 02_preprocessing.md
│   ├── 03_exploratory_data_analysis.md
│   ├── 04_regression_models.md
│   └── 05_classification_models.md
└── figures/                                 14 figures extracted from the paper
    ├── fig01_ln_ic50_histogram.png
    ├── fig02_ln_ic50_qq_plot.png
    ├── fig03_driver_gene_mutation_frequencies.png
    ├── fig04_mutation_lnic50_correlations.png
    ├── fig05_pathway_enrichment_sensitive_vs_resistant.png
    ├── fig06_top5_effective_drugs.png
    ├── fig07_regression_top20_features_no_mut_no_pca.png
    ├── fig08_regression_top20_features_with_mut_no_pca.png
    ├── fig09_classifier_roc_no_mutation.png
    ├── fig10_classifier_confusion_matrix_no_mutation.png
    ├── fig11_classifier_top20_features_no_mutation.png
    ├── fig12_classifier_roc_with_mutation.png
    ├── fig13_classifier_confusion_matrix_with_mutation.png
    └── fig14_classifier_top20_features_with_mutation.png
```

## Reproduce the analysis

The scripts use relative paths and do not contain machine-specific locations. Put
the five original GDSC downloads in `data/raw/`, then run from the repository root:

```bash
pip install -r requirements.txt
python src/01_build_dataset.py
python src/02_exploratory_analysis.py
python src/03_classification_no_pca.py
python src/04_classification_pca.py
python src/05_regression_no_pca.py
python src/06_regression_pca.py
python src/07_classification_tuning.py
```

Raw and processed datasets are intentionally excluded from version control. Each
script accepts `--help` and supports custom input/output paths.

The published metrics reproduce the original response-level 80/20 split. This
tests held-out cell-line/drug response pairs when the same cell lines and drugs may
also occur in training. Generalization to entirely unseen cell lines or unseen
drugs requires a grouped split by `COSMIC_ID` or `DRUG_ID`, respectively.

### V2: leakage-aware validation

The V2 implements those grouped tests, fits preprocessing on training data only,
removes response-derived predictors, audits group overlap, and reports uncertainty
across repeated holdouts:

```bash
python src/08_v2_grouped_validation.py --repeats 5
```

See [`docs/V2_VALIDATION.md`](docs/V2_VALIDATION.md) for the rationale and
interpretation guide. Until the original GDSC files are rerun, grouped metrics are
intentionally not claimed here.

---

## Key findings

**Biological signal — feature importance across models**

- **RNA polymerase as drug target** dominates the classifier — LUAD cell lines
  respond very differently to drugs targeting the transcriptional machinery,
  consistent with the literature on transcription as a tumor vulnerability
  (Ferreira et al., 2020; Saproo et al., 2023).
- **Romidepsin (HDAC inhibitor)** showed the lowest mean LN_IC50 across the 286
  screened compounds — a drug-repurposing candidate for LUAD given the
  pathway-level enrichment of chromatin/histone acetylation in the sensitive
  subset.
- **BCL-2 / MCL-1 family** features (apoptosis evasion) surface as top predictors
  once mutation data is added — clinically aligned with BH3-mimetic strategies in
  LUAD.

**Mutation-LN_IC50 point-biserial correlations**

- Negative (more sensitive when mutated): **STK11**, **KEAP1**, **GRIN2A**
  (STK11 strongest at ≈ −0.07).
- Positive (more resistant when mutated): **ASXL2**, **PALB2**, **PTPN13**,
  **PRKAR1A**.

**Methodological takeaways**

- Gene expression alone is the dominant predictive signal; somatic mutation status
  adds marginal performance but meaningful biological interpretability.
- PCA improves runtime on a 17 k-feature space but **degrades classifier
  performance** and obscures biological feature importance — kept only for the
  regression speedup case.
- Lasso (R² = 0.68) underperforms XGBoost — confirms non-linear interactions
  dominate the genomic feature space.

For the full results, figures, and discussion, see
[`docs/Capstone_Paper.pdf`](docs/Capstone_Paper.pdf).

---

## Methodology breakdown

The `methodology/` directory documents the analytical pipeline step by step. Each
file is self-contained and references the corresponding section of the paper:

| Step                                          | Doc                                                                    | Paper section |
|-----------------------------------------------|------------------------------------------------------------------------|---------------|
| 1. Dataset ingestion + LUAD filtering         | [`methodology/01_data_ingestion.md`](methodology/01_data_ingestion.md) | § 2.1–2.2     |
| 2. Preprocessing + feature engineering        | [`methodology/02_preprocessing.md`](methodology/02_preprocessing.md)   | § 2.3         |
| 3. Exploratory data analysis                  | [`methodology/03_exploratory_data_analysis.md`](methodology/03_exploratory_data_analysis.md) | § 3 |
| 4. XGBoost regression — 4 configurations      | [`methodology/04_regression_models.md`](methodology/04_regression_models.md) | § 4.1 |
| 5. XGBoost classification — 4 configurations  | [`methodology/05_classification_models.md`](methodology/05_classification_models.md) | § 4.2 |

---

## Tools used

| Stage                       | Tool                                                                   |
|-----------------------------|------------------------------------------------------------------------|
| Language                    | Python 3                                                               |
| Data handling               | pandas · NumPy                                                         |
| Preprocessing               | scikit-learn (`StandardScaler`, `OneHotEncoder`, `PCA`)                |
| Modeling — regression       | XGBoost (`XGBRegressor`)                                               |
| Modeling — classification   | XGBoost (`XGBClassifier`)                                              |
| Baseline                    | scikit-learn (`Lasso`)                                                 |
| Statistics                  | SciPy (`stats.pointbiserialr`, `stats.shapiro`)                        |
| Visualization               | matplotlib · seaborn                                                   |

Install pinned dependencies via:

```bash
pip install -r requirements.txt
```

---

## Authors

| Name             | Role                                  |
|------------------|---------------------------------------|
| Jordan Dautelle  | Author (corresponding)                |
| Hang Tran        | Author                                |
| David Yoon       | Author                                |
| Alli Warren      | Author                                |
| Dr. Ken Sweat    | Advisor (Arizona State University)    |

---

## Acknowledgements

Thanks to the **Genomics of Drug Sensitivity in Cancer** project at the Wellcome
Sanger Institute and the Cancer Genome Project for making the raw multi-omic data
publicly available, and to **Dr. Ken Sweat** for capstone supervision.

---

## License

MIT — see [`LICENSE`](LICENSE).
