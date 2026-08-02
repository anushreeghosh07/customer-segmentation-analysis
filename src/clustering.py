"""
clustering.py
--------------
Clustering utilities for the segmentation project.

Two clustering approaches are implemented deliberately, as a
methodological comparison rather than a "try everything" exercise:

    1. K-Means on one-hot-encoded + scaled data
       -> a common naive baseline. Euclidean distance on one-hot
          categorical vectors is not really meaningful, so this is
          included specifically to demonstrate the limitation, not
          because it's the recommended approach.

    2. K-Prototypes on raw mixed-type data
       -> combines Euclidean distance (numeric) with a matching
          dissimilarity measure (categorical), which is the
          methodologically correct approach for this dataset.

Also included: PCA for visualization, elbow/silhouette helpers for
choosing k, and cluster profiling utilities.
"""

import numpy as np
import pandas as pd
import gower
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from kmodes.kprototypes import KPrototypes

from .data_loader import CATEGORICAL_COLS, NUMERIC_COLS


def prepare_kmeans_input(df_raw: pd.DataFrame) -> np.ndarray:
    """
    One-hot encode categoricals + standardize numerics, for the
    K-Means baseline. Returns a scaled numpy array.
    """
    numeric = df_raw[NUMERIC_COLS]
    categorical = pd.get_dummies(df_raw[CATEGORICAL_COLS].astype(str), drop_first=False)
    combined = pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
    scaled = StandardScaler().fit_transform(combined)
    return scaled, combined.columns.tolist()


def prepare_kprototypes_input(df_raw: pd.DataFrame):
    """
    Prepares data for K-Prototypes: numeric columns standardized,
    categorical columns kept as-is (integer codes), with their
    column indices flagged for the algorithm.
    """
    df = df_raw[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    df[NUMERIC_COLS] = StandardScaler().fit_transform(df[NUMERIC_COLS])
    categorical_idx = [df.columns.get_loc(c) for c in CATEGORICAL_COLS]
    return df.to_numpy(), categorical_idx, df.columns.tolist()


def elbow_and_silhouette_kmeans(X: np.ndarray, k_range=range(2, 9)) -> pd.DataFrame:
    """Computes inertia (elbow) and silhouette score across a range of k for K-Means."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
        })
    return pd.DataFrame(rows)


def cost_kprototypes(X: np.ndarray, categorical_idx: list, k_range=range(2, 9)) -> pd.DataFrame:
    """Computes K-Prototypes cost (elbow-equivalent) across a range of k."""
    rows = []
    for k in k_range:
        kp = KPrototypes(n_clusters=k, init="Cao", random_state=42, n_init=5, verbose=0)
        kp.fit_predict(X, categorical=categorical_idx)
        rows.append({"k": k, "cost": kp.cost_})
    return pd.DataFrame(rows)


def elbow_and_silhouette_kprototypes(X: np.ndarray, categorical_idx: list, gower_matrix: np.ndarray,
                                      k_range=range(2, 9)) -> pd.DataFrame:
    """
    Computes K-Prototypes cost (elbow) AND Gower-distance silhouette
    across a range of k, mirroring elbow_and_silhouette_kmeans so both
    methods' choice of k can be justified and compared the same way.
    K-Prototypes has no native Euclidean silhouette (mixed distance),
    so Gower silhouette is used as the fair, method-agnostic yardstick.
    """
    rows = []
    for k in k_range:
        kp = KPrototypes(n_clusters=k, init="Cao", random_state=42, n_init=5, verbose=0)
        labels = kp.fit_predict(X, categorical=categorical_idx)
        rows.append({
            "k": k,
            "cost": kp.cost_,
            "silhouette": silhouette_score(gower_matrix, labels, metric="precomputed"),
        })
    return pd.DataFrame(rows)


def elbow_and_silhouette_hierarchical(Z: np.ndarray, gower_matrix: np.ndarray, k_range=range(2, 9)) -> pd.DataFrame:
    """
    Hierarchical clustering has no direct 'elbow' (no iterative cost
    function), but cutting the same tree at different k and scoring each
    cut's Gower silhouette gives an equivalent systematic way to justify
    k, on top of visually reading the dendrogram merge heights.
    """
    rows = []
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        rows.append({
            "k": k,
            "silhouette": silhouette_score(gower_matrix, labels, metric="precomputed"),
        })
    return pd.DataFrame(rows)


def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X)


def run_kprototypes(X: np.ndarray, categorical_idx: list, k: int):
    kp = KPrototypes(n_clusters=k, init="Cao", random_state=42, n_init=5, verbose=0)
    labels = kp.fit_predict(X, categorical=categorical_idx)
    return labels, kp


def compare_clusterings(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Adjusted Rand Index between two sets of cluster labels for the same rows."""
    ari = adjusted_rand_score(labels_a, labels_b)
    return {
        "adjusted_rand_index": ari,
        "interpretation": (
            "1.0 = identical partitions, 0.0 = agreement no better than random. "
            f"Observed ARI = {ari:.3f}."
        ),
    }


def pca_2d(X: np.ndarray) -> tuple:
    """Reduces X to 2 components for visualization. Returns (transformed, explained_variance_ratio)."""
    pca = PCA(n_components=2, random_state=42)
    transformed = pca.fit_transform(X)
    return transformed, pca.explained_variance_ratio_


def profile_clusters(df_labeled: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Builds a per-cluster profile table: mean of numeric vars,
    mode (+ % share) of categorical vars, and cluster size.
    Expects df_labeled to have human-readable category values.
    """
    df = df_labeled.copy()
    df["Cluster"] = cluster_labels

    profiles = []
    for cluster_id, group in df.groupby("Cluster"):
        profile = {"Cluster": cluster_id, "Size": len(group), "Share (%)": round(100 * len(group) / len(df), 1)}
        for col in NUMERIC_COLS:
            profile[f"{col} (mean)"] = round(group[col].mean(), 1)
        for col in CATEGORICAL_COLS:
            mode_val = group[col].mode().iloc[0]
            share = round(100 * (group[col] == mode_val).mean(), 1)
            profile[col] = f"{mode_val} ({share}%)"
        profiles.append(profile)

    return pd.DataFrame(profiles)


# ---------------------------------------------------------------------------
# Gower distance utilities
# ---------------------------------------------------------------------------
# Gower distance combines numeric and categorical variables into a single,
# principled dissimilarity measure (unlike Euclidean distance on one-hot
# data). It is used two ways below:
#   1. As the distance matrix that hierarchical clustering is built on.
#   2. As a *common, fair yardstick* to empirically compare K-Means and
#      K-Prototypes: since neither algorithm was directly optimized to
#      minimize Gower distance, scoring both on it tells us which one
#      actually captures the mixed-type structure better in practice,
#      not just in theory.

def compute_gower_matrix(df_raw: pd.DataFrame) -> np.ndarray:
    """Computes the full pairwise Gower distance matrix for the mixed-type data."""
    df = df_raw[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    # the gower package requires numeric columns as float (not int) internally
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)
    cat_features = [False] * len(NUMERIC_COLS) + [True] * len(CATEGORICAL_COLS)
    return gower.gower_matrix(df, cat_features=cat_features)


def gower_silhouette(gower_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette score computed on the Gower distance matrix for a given
    set of cluster labels. This works for labels from ANY clustering
    method (K-Means, K-Prototypes, hierarchical) since it only needs
    the labels and a precomputed distance matrix -- making it the
    common ground for a fair empirical comparison across methods.
    """
    return silhouette_score(gower_matrix, labels, metric="precomputed")


def categorical_purity(df_raw: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Average, size-weighted 'purity' of categorical variables within
    clusters: for each categorical column and each cluster, what
    fraction of members share the most common (modal) value, averaged
    across clusters (weighted by cluster size) and across columns.
    Higher = clusters are more internally homogeneous on categorical
    variables, which is exactly what a mixed-type-aware method like
    K-Prototypes should improve on relative to a naive K-Means baseline.
    """
    df = df_raw[CATEGORICAL_COLS].copy()
    df["Cluster"] = labels
    n = len(df)

    col_purities = []
    for col in CATEGORICAL_COLS:
        weighted_purity = 0.0
        for _, group in df.groupby("Cluster"):
            mode_share = group[col].value_counts(normalize=True).iloc[0]
            weighted_purity += mode_share * (len(group) / n)
        col_purities.append(weighted_purity)

    return float(np.mean(col_purities))


def numeric_separation(df_raw: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    One-way ANOVA F-statistic for Age and Income across clusters.
    A higher F means the clustering differentiates groups more strongly
    on that numeric variable. This is the most decisive result-based
    metric for comparing K-Means vs K-Prototypes here: K-Means one-hot
    encodes 5 categorical columns into 14 dummy columns vs only 2 numeric
    columns, so Euclidean distance ends up dominated by categorical
    dimensionality -- K-Means clusters can end up looking clean on
    categorical variables while barely differentiating Age/Income at all.
    A mixed-type-aware method should show materially higher F-statistics
    here if it's genuinely using the numeric information better, not just
    performing similarly in theory.
    """
    from scipy import stats
    groups = np.unique(labels)
    age_groups = [df_raw.loc[labels == c, "Age"] for c in groups]
    income_groups = [df_raw.loc[labels == c, "Income"] for c in groups]
    f_age, p_age = stats.f_oneway(*age_groups)
    f_income, p_income = stats.f_oneway(*income_groups)
    return {"age_f": f_age, "age_p": p_age, "income_f": f_income, "income_p": p_income}


def empirical_comparison(df_raw: pd.DataFrame, gower_matrix: np.ndarray, labels_dict: dict) -> pd.DataFrame:
    """
    Builds a results table comparing multiple clustering label sets on
    THREE result-based (not theory-based) metrics:
      - Gower silhouette: overall geometric cluster separation on a
        mixed-type-fair distance.
      - Categorical purity: how homogeneous clusters are on categorical
        variables.
      - Age/Income ANOVA F-statistic: how strongly clusters differentiate
        on the numeric variables -- often the most business-relevant and
        most revealing metric, since it's the one most likely to expose
        whether a method is actually using numeric information or being
        swamped by categorical dimensionality.
    labels_dict maps a method name (str) to its label array.
    """
    rows = []
    for method_name, labels in labels_dict.items():
        sep = numeric_separation(df_raw, labels)
        rows.append({
            "Method": method_name,
            "Gower silhouette": round(gower_silhouette(gower_matrix, labels), 4),
            "Categorical purity": round(categorical_purity(df_raw, labels), 4),
            "Age ANOVA F": round(sep["age_f"], 1),
            "Income ANOVA F": round(sep["income_f"], 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hierarchical clustering (Gower distance + linkage)
# ---------------------------------------------------------------------------

def hierarchical_linkage(gower_matrix: np.ndarray, method: str = "average"):
    """
    Builds a hierarchical clustering linkage matrix from a precomputed
    Gower distance matrix. Note: Ward's linkage formally requires
    Euclidean distance, so for a Gower-based (non-Euclidean) distance
    matrix, 'average' or 'complete' linkage is the methodologically
    appropriate choice -- we compare both explicitly below rather than
    defaulting to Ward out of habit.
    """
    condensed = squareform(gower_matrix, checks=False)
    return linkage(condensed, method=method)


def cophenetic_correlation(gower_matrix: np.ndarray, Z: np.ndarray) -> float:
    """How well the dendrogram's merge distances preserve the original
    pairwise Gower distances. Closer to 1 is better. This diagnostic has
    no equivalent in K-Means/K-Prototypes -- part of why hierarchical
    clustering is worth including as a third, independently-validated method."""
    condensed = squareform(gower_matrix, checks=False)
    c, _ = cophenet(Z, condensed)  # cophenet() returns (correlation_coefficient, coph_dists) when given Y
    return float(c)


def cut_tree(Z: np.ndarray, k: int) -> np.ndarray:
    """Cuts a hierarchical linkage tree into k flat clusters (1-indexed labels)."""
    return fcluster(Z, t=k, criterion="maxclust")
