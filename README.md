# Customer Segmentation Analysis

Segmenting 2,000 supermarket loyalty-card customers into actionable groups using three clustering methods, backed by formal hypothesis testing — and an empirical, results-based justification for which method actually performs best, not just a theoretical one.

## Why this project

Most segmentation tutorials run K-Means and stop there. This project instead treats **method choice as a first-class question**: the dataset has mixed numeric and categorical variables, so it compares a naive K-Means baseline (one-hot encoded categoricals, Euclidean distance) against K-Prototypes and hierarchical clustering (both built on distance measures designed for mixed data) — and rather than assuming the "correct" method wins, it tests that empirically. The result is a genuine finding, not a foregone conclusion: K-Means' one-hot representation is dominated by categorical dimensionality (14 dummy columns vs. 2 numeric ones), so it barely differentiates customers by Income (ANOVA F=112) despite Income being a variable it was explicitly given — while K-Prototypes produces clusters ~6x more differentiated on Income (F=686), independently corroborated by hierarchical clustering.


## Dataset

2,000 anonymized customers from an FMCG store, collected via loyalty cards. 7 features: `Sex`, `Marital status`, `Age`, `Education`, `Income`, `Occupation`, `Settlement size`. No missing values. Full variable dictionary in [`data/segmentation_data_legend.xlsx`](data/segmentation_data_legend.xlsx).

## What's inside

| Step | What it does |
|---|---|
| EDA | Distributions, income-by-education, age-vs-income relationships |
| Hypothesis testing | Chi-square, t-test/Mann-Whitney, ANOVA/Kruskal-Wallis, Pearson correlation — each with automatic assumption checking (normality, equal variance) and fallback to the correct non-parametric test |
| K-Means | Naive baseline on one-hot encoded data; k chosen via elbow + silhouette |
| K-Prototypes | Mixed-type clustering (Huang, 1998); k chosen via elbow (cost) + Gower-distance silhouette |
| K-Means vs. K-Prototypes | Empirical (not just theoretical) comparison: Adjusted Rand Index, Gower silhouette, categorical purity, and Age/Income ANOVA F-statistics |
| Hierarchical clustering | Gower distance + linkage comparison (average/complete/ward), cophenetic correlation, dendrogram, silhouette-based k selection |
| Three-way comparison | All pairwise ARI + empirical metrics across K-Means, K-Prototypes, and Hierarchical |
| Profiling | Per-cluster summary stats translated into business personas |
| PCA | 2D projection for cluster visualization across all three methods + multicollinearity note |
| Report | Findings, recommended method, limitations, next steps |

## Key results

- Income and Age are both significantly associated with Education, Occupation, Marital status, and Sex (all p < 0.001).
- **K-Means' Gower silhouette score is actually higher than K-Prototypes'** at first glance — but this turns out to be a dimensionality artifact: one-hot encoding expands 5 categorical columns into 14 dimensions vs. only 2 numeric ones, so K-Means ends up clustering almost entirely by categorical combinations.
- The decisive result-based metric is numeric separation: K-Prototypes produces clusters with a **~6x higher Income ANOVA F-statistic** (686 vs. 112) and ~1.5x higher Age F-statistic than K-Means — concrete evidence, not just theory, that K-Prototypes makes far better use of the numeric variables that matter most for a business segmentation.
- Hierarchical clustering (Gower distance, average linkage) independently corroborates this: it agrees with K-Prototypes more than either agrees with the K-Means baseline (pairwise Adjusted Rand Index), and shows similarly strong numeric separation.
- Four interpretable segments emerge from K-Prototypes, differentiated by age, income, education, and settlement size — see the notebook for full profiles and proposed personas.

## Project structure

```
customer-segmentation-project/
├── data/
│   ├── segmentation_data.csv
│   └── segmentation_data_legend.xlsx
├── src/
│   ├── data_loader.py       # loading + label mapping
│   ├── stats_tests.py       # hypothesis test wrappers with assumption checks
│   └── clustering.py        # K-Means, K-Prototypes, hierarchical (Gower), PCA, empirical comparison, profiling
├── notebooks/
│   └── customer_segmentation_analysis.ipynb   # full narrative analysis (Python)
├── outputs/
│   └── figures/              # exported plots
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/customer_segmentation_analysis.ipynb
```

## Limitations & next steps

- No purchase/spend variable exists in this dataset, so segments can't be directly tied to revenue without additional data.
- Cluster count (k=4) was chosen via elbow/silhouette diagnostics but involves some judgment.
- Natural extensions: Gower distance + hierarchical clustering as a third comparison method; incorporating spend data for an RFM-style segmentation; deploying the segment classifier as an API.
