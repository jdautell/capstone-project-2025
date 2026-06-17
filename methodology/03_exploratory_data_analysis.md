# Step 3 — Exploratory Data Analysis

> Maps to **§ 3** of [`docs/Capstone_Paper.pdf`](../docs/Capstone_Paper.pdf).

The EDA stage characterizes the LUAD-restricted multi-omic table before any
modeling. All EDA figures are reproduced in the paper PDF.

## 3.1 Distribution of LN_IC50 (Figures 1–2 in the paper)

The log-transformed IC50 target follows a **single-peaked, slightly
left-skewed** distribution centered around 4. A **Q-Q plot vs. the standard
normal** shows deviations in the tails; the **Shapiro–Wilk** statistic is
**0.9554**, close to 1 but not perfectly normal.

Practical consequences:

- Justifies using **tree-based models (XGBoost)** that do not assume normality.
- Justifies binarizing at **LN_IC50 = 0** for the classification framing —
  the threshold captures the most-sensitive tail without sitting on the dense
  central mode.

## 3.2 Driver-gene mutation frequencies (Figure 3)

Frequencies of the top driver-gene mutations in the LUAD cell-line panel:

| Gene      | Approximate frequency | Biological note                                              |
|-----------|-----------------------|--------------------------------------------------------------|
| **TP53**  | ~81 %                 | Genomic instability, poor prognosis (Li et al., 2023)        |
| **KRAS**  | ~34 %                 | Drug resistance and aggressive tumor biology (Jones et al., 2021) |
| **STK11** | 15–25 %               | Immune evasion, reduced ICI efficacy (Kwack et al., 2020)    |
| **KEAP1** | 15–25 %               | Oxidative stress response, drug resistance (Yu & Xiao, 2021) |
| **SMARCA4, LRP1B, FAT3, EGFR, HLA-A, CDKN2A** | additional driver hits previously reported in LUAD | |

Frequencies sum to more than 100 % because a single cell line can carry mutations
in multiple driver genes.

## 3.3 Mutation × LN_IC50 — point-biserial correlation (Figure 4)

For each driver gene, a **point-biserial correlation** between binary mutation
status and continuous LN_IC50 was computed (appropriate test for a binary × continuous
pair):

```
from scipy.stats import pointbiserialr
r, p = pointbiserialr(df[gene], df["LN_IC50"])
```

| Direction         | Top genes                                  | Interpretation                                       |
|-------------------|--------------------------------------------|------------------------------------------------------|
| **Negative** (sensitive when mutated) | **STK11**, **KEAP1**, **GRIN2A** (STK11 strongest at ≈ −0.07) | Metabolic / oxidative vulnerabilities |
| **Positive** (resistant when mutated) | **ASXL2**, **PALB2**, **PTPN13**, **PRKAR1A** (≈ +0.10) | Putative resistance mechanisms |

These directions are consistent with the literature for STK11 (Krall et al., 2017)
and identify candidate biomarkers for treatment stratification.

## 3.4 Pathway-level enrichment (Figure 5)

Mean LN_IC50 was computed per drug **target pathway**, then split into a
**sensitive** subset (mean LN_IC50 < 0) and a **resistant** subset
(mean LN_IC50 > 0).

| Subset      | Top enriched pathways                                                   |
|-------------|-------------------------------------------------------------------------|
| Sensitive   | Chromatin histone acetylation, Metabolism, Mitosis                       |
| Resistant   | Protein stability & degradation, Hormone-related, Genome integrity       |

The chromatin/histone-acetylation enrichment in sensitive lines previews the
Romidepsin (HDAC inhibitor) result in § 3.5.

## 3.5 Drug-level efficacy (Figure 6)

Across the **286** GDSC2 compounds, the lowest-mean-LN_IC50 (most potent on LUAD
cell lines) are:

1. **Romidepsin** — FDA-approved HDAC inhibitor (Grant et al., 2010)
2. **Sepantronium bromide (YM155)**
3. **Bortezomib**
4. **Dactinomycin**
5. **SN-38**

Romidepsin is not standard-of-care for LUAD, but its consistently low LN_IC50
across LUAD cell lines — paired with the pathway-level enrichment in
chromatin/histone acetylation among sensitive lines — flags it as a credible
**drug-repurposing candidate**, consistent with literature on histone-acetylation
modulators in lung cancer (Falkenberg & Johnstone, 2014; Natu et al., 2024).

---

⬅️ Previous: [`02_preprocessing.md`](02_preprocessing.md)
➡️ Next: [`04_regression_models.md`](04_regression_models.md)
