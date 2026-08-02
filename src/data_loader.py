"""
data_loader.py
---------------
Loads and prepares the FMCG customer segmentation dataset.

The raw file stores categorical variables as integer codes (per the
data legend). This module attaches human-readable labels and exposes
both a "raw" (numeric-coded) and "labeled" version of the dataframe,
since different steps of the analysis need different representations:
    - clustering algorithms need numeric codes
    - EDA plots and profiling tables are far more readable with labels
"""

import pandas as pd

SEX_MAP = {0: "male", 1: "female"}
MARITAL_MAP = {0: "single", 1: "non-single"}
EDUCATION_MAP = {0: "other/unknown", 1: "high school", 2: "university", 3: "graduate school"}
OCCUPATION_MAP = {0: "unemployed/unskilled", 1: "skilled employee", 2: "management/highly qualified"}
SETTLEMENT_MAP = {0: "small city", 1: "mid-sized city", 2: "big city"}

CATEGORICAL_COLS = ["Sex", "Marital status", "Education", "Occupation", "Settlement size"]
NUMERIC_COLS = ["Age", "Income"]


def load_raw(path: str) -> pd.DataFrame:
    """Load the dataset exactly as stored (integer-coded categoricals)."""
    df = pd.read_csv(path)
    return df


def load_labeled(path: str) -> pd.DataFrame:
    """Load the dataset with categorical codes replaced by readable labels."""
    df = load_raw(path)
    df = df.copy()
    df["Sex"] = df["Sex"].map(SEX_MAP)
    df["Marital status"] = df["Marital status"].map(MARITAL_MAP)
    df["Education"] = df["Education"].map(EDUCATION_MAP)
    df["Occupation"] = df["Occupation"].map(OCCUPATION_MAP)
    df["Settlement size"] = df["Settlement size"].map(SETTLEMENT_MAP)
    return df


def basic_quality_report(df: pd.DataFrame) -> dict:
    """Quick data-quality summary: shape, dtypes, missing values, duplicates."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_ids": int(df["ID"].duplicated().sum()) if "ID" in df.columns else None,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
