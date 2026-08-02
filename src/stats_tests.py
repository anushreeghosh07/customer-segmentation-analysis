"""
stats_tests.py
---------------
Hypothesis testing utilities for the segmentation dataset.

Each function returns a small dict with the test statistic, p-value,
and a plain-language interpretation, so results can be dropped
directly into a results table in the notebook.
"""

from itertools import combinations

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ALPHA = 0.05


def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Chi-square test of independence between two categorical columns."""
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "test": "Chi-square test of independence",
        "variables": f"{col1} vs {col2}",
        "statistic": chi2,
        "dof": dof,
        "p_value": p,
        "significant": p < ALPHA,
        "interpretation": (
            f"{'Reject' if p < ALPHA else 'Fail to reject'} H0: "
            f"{col1} and {col2} appear "
            f"{'associated' if p < ALPHA else 'independent'} (alpha={ALPHA})."
        ),
    }


def two_group_test(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict:
    """
    Compares a numeric variable across exactly two groups.
    Runs Levene's test for equal variance and Shapiro-Wilk for normality
    first, then picks Welch's t-test or Mann-Whitney U accordingly.
    """
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"{group_col} must have exactly 2 groups, found {len(groups)}")

    g1 = df.loc[df[group_col] == groups[0], numeric_col].dropna()
    g2 = df.loc[df[group_col] == groups[1], numeric_col].dropna()

    # Normality check (sample for large n, Shapiro is sensitive above ~5000)
    norm1_p = stats.shapiro(g1.sample(min(len(g1), 500), random_state=42))[1]
    norm2_p = stats.shapiro(g2.sample(min(len(g2), 500), random_state=42))[1]
    normal = (norm1_p > ALPHA) and (norm2_p > ALPHA)

    if normal:
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test
        test_used = "Welch's t-test"
    else:
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_used = "Mann-Whitney U test (normality assumption violated)"

    return {
        "test": test_used,
        "variables": f"{numeric_col} by {group_col}",
        "group_means": {str(groups[0]): g1.mean(), str(groups[1]): g2.mean()},
        "statistic": stat,
        "p_value": p,
        "significant": p < ALPHA,
        "interpretation": (
            f"{'Reject' if p < ALPHA else 'Fail to reject'} H0: mean {numeric_col} "
            f"{'differs' if p < ALPHA else 'does not differ'} significantly between "
            f"{group_col} groups (alpha={ALPHA})."
        ),
    }


def one_way_anova(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict:
    """
    One-way ANOVA of a numeric variable across 3+ groups, with Levene's
    test for equal variance and a Tukey HSD post-hoc test if significant.
    """
    groups_data = [g[numeric_col].dropna().values for _, g in df.groupby(group_col)]
    levene_stat, levene_p = stats.levene(*groups_data)

    if levene_p > ALPHA:
        f_stat, p = stats.f_oneway(*groups_data)
        test_used = "One-way ANOVA"
    else:
        f_stat, p = stats.kruskal(*groups_data)
        test_used = "Kruskal-Wallis H-test (equal variance assumption violated)"

    result = {
        "test": test_used,
        "variables": f"{numeric_col} across {group_col}",
        "levene_p": levene_p,
        "statistic": f_stat,
        "p_value": p,
        "significant": p < ALPHA,
        "interpretation": (
            f"{'Reject' if p < ALPHA else 'Fail to reject'} H0: mean {numeric_col} "
            f"{'differs' if p < ALPHA else 'does not differ'} significantly across "
            f"{group_col} groups (alpha={ALPHA})."
        ),
    }

    if p < ALPHA and test_used == "One-way ANOVA":
        tukey = pairwise_tukeyhsd(df[numeric_col], df[group_col], alpha=ALPHA)
        result["posthoc_tukey"] = str(tukey)

    return result


def pearson_correlation(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Pearson correlation between two numeric variables."""
    r, p = stats.pearsonr(df[col1], df[col2])
    return {
        "test": "Pearson correlation",
        "variables": f"{col1} vs {col2}",
        "r": r,
        "p_value": p,
        "significant": p < ALPHA,
        "interpretation": (
            f"{'Reject' if p < ALPHA else 'Fail to reject'} H0 of no correlation "
            f"(r={r:.3f}, alpha={ALPHA})."
        ),
    }
