# capstone-project-2025
Predicting drug sensitivity in LUAD using machine learning and multi-omic data.

# Predicting Drug Sensitivity in Lung Adenocarcinoma Using Machine Learning

### Jordan Dautelle, Hang Tran, David Yoon, Alli Warren

## Overview

This project explores the use of machine learning models to predict the sensitivity of Lung Adenocarcinoma (LUAD) cancer cell lines to various anti-cancer drugs. Lung cancer is the most lethal cancer globally, and patients often respond differently to treatments. By leveraging multi-omic data—specifically gene expression and somatic mutation profiles from the Genomics of Drug Sensitivity in Cancer (GDSC) project—we developed predictive models to identify intricate relationships between molecular features and drug sensitivity. The goal is to contribute to precision oncology by guiding more personalized treatment strategies.

## Table of Contents

* [Project Goal](#project-goal)
* [Dataset Sources](#dataset-sources)
* [Methods](#methods)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Machine Learning Models and Results](#machine-learning-models-and-results)
* [Conclusion & Future Work](#conclusion--future-work)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)
* [License](#license)

## Project Goal

The primary objective was to predict drug sensitivity in LUAD cell lines using machine learning models. We considered two types of models:

1.  **Regression Models:** To predict the natural logarithm of half-maximal inhibitory concentration ($LN\_IC50$) values, which is a standard measure for drug sensitivity.

2.  **Classification Models:** To categorize samples as either "sensitive" or "resistant" based on a threshold of $LN\_IC50$.

We also sought to determine if incorporating somatic mutation data would improve predictive performance or biological interpretability.

## Dataset Sources

This project utilized publicly available datasets from the Genomics of Drug Sensitivity in Cancer (GDSC) project. The following datasets were merged for our analysis:

* `GDSC2_fitted_dose_response_27Oct23`: Contains drug sensitivity measures (specifically, LN_IC50 values).

* `Cell_Lines_Details`: Provides metadata for each cell line, used to filter for LUAD-specific samples.

* `Cell_line_RMA_proc_basalExp`: A matrix of gene expression values for thousands of genes.

* `mutations_all 20230202`: Catalogs binary mutation statuses for known cancer driver genes.

* `screened_compounds_rel_8.5`: Provides information on all screened compounds, including drug names and target pathways.

## Methods

Our analysis was conducted using Python 3 and followed a multi-step process:

1.  **Data Merging:** Datasets were merged using unique identifiers like COSMIC ID and DRUG_ID to create a comprehensive multi-omic resource.

2.  **Preprocessing:**

    * Missing values were handled.

    * Categorical features were transformed using one-hot encoding.

    * Continuous features were standardized using `StandardScaler` from the `scikit-learn` library.

3.  **Feature Engineering:**

    * Principal Component Analysis (PCA) was applied in selected model pipelines to reduce dimensionality and computational cost.

    * For our classification models, LN_IC50 values were binarized with a threshold of $LN\_IC50<0$ for the "sensitive" class and $LN\_IC50\ge0$ for the "resistant" class.

## Exploratory Data Analysis (EDA)

The EDA revealed several key insights into the dataset and the biological context of LUAD:

* **LN_IC50 Distribution:** The log-transformed IC50 values followed a slightly left-skewed distribution, justifying the use of models like XGBoost that are robust to non-normal data.

* **Driver Gene Mutations:** TP53 was the most frequently mutated driver gene (~81%), consistent with existing literature. Other frequent mutations included KRAS and STK11.

* **Mutation-Drug Sensitivity Correlation:** A point-biserial correlation analysis showed negative correlations between mutations in genes like STK11 and KEAP1 and LN_IC50 values, suggesting increased drug sensitivity. Conversely, mutations in genes like ASXL2 showed a positive correlation, potentially indicating drug resistance.

* **Pathway Enrichment:** Analysis of the most enriched pathways revealed that **Chromatin histone acetylation** and **Metabolism** were highly active in sensitive cell lines, while **Protein stability and degradation** pathways were more prominent in resistant cell lines.

* **Drug Efficacy:** Of the 286 drugs screened, **Romidepsin** was identified as the most effective compound in LUAD cell lines based on its average LN_IC50 value.

## Machine Learning Models and Results

We primarily used XGBoost for both regression and classification tasks, comparing model performance with and without PCA and mutation data.

### Regression Models

* The model incorporating both PCA and mutation data achieved the best performance with an $R^{2}$ **of 0.7976**, **RMSE of 1.2316**, and **MAE of 0.9321**.

* The non-PCA models, while slightly less performant, provided the most biologically interpretable results, highlighting RNA polymerase-related targets and specific drugs (like Sepantronium bromide) as top predictors.

### Classification Models

* Classification models consistently achieved high performance with an **ROC AUC of ~0.97** and **accuracy of ~96%**.

* The model without mutation data identified **RNA polymerase** as the most significant feature.

* Adding mutation data did not substantially improve performance but introduced key biological features, such as the **BCL-2/MCL-1 family** of genes, into the top predictors. This highlights the value of mutation data for biological insight.

### Key Takeaway

Our findings validate the trade-off between model performance and interpretability. While PCA improves computational speed for high-dimensional data, it can obscure crucial biological insights. Gene expression proved to be a stronger predictive signal than mutation status, but mutation data added significant value for biological interpretation.

## Conclusion & Future Work

Our project successfully demonstrates the utility of machine learning in predicting drug sensitivity in LUAD. The models successfully identified biologically and clinically relevant features, such as RNA polymerase targets and BCL-2 family genes, which align with our understanding of cancer biology.

**Future work includes:**

* Expanding the analysis to include more cancer types and datasets (e.g., proteomics or epigenomics).

* Validating the models on external datasets to confirm their reliability.

* Exploring alternative machine learning methods like deep learning or graph neural networks.


## Authors

* Jordan Dautelle

* Hang Tran

* David Yoon

* Alli Warren

## Acknowledgements

* **Dr. Ken Sweat**, for guidance and support on the project.

* **Arizona State University**, for providing resources for this capstone.

## License

This project is licensed under the MIT License.
