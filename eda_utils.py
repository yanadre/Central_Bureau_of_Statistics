"""Utility functions for the CBS exploratory data analysis notebook."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


STOP_WORDS = ["", "nan", "none", "null", 'כנ"ל']

DEFAULT_ID_COL = "ID"
DEFAULT_TARGET_COL = "SemelMishlachSofi"
# DEFAULT_INDUSTRY_COL = "SemelAnafSofi"

# DEFAULT_TEXT_COLS = [
#     "ShemAvoda",
#     "SugAvoda",
#     "ShemMachlaka",
#     "SugMachlaka",
#     "EzoAvoda",
#     "TeurPeula",
#     "TeurTafkid",
# ]

DEFAULT_TEXT_ORDER_FOR_EMBEDDINGS = [
    "EzoAvoda",
    "SugAvoda",
    "TeurPeula",
    "TeurTafkid",
    "ShemMachlaka",
    "SugMachlaka",
    "ShemAvoda",
]


# def load_survey_data(data_path, sheet_name="Survey Dataset", id_col=DEFAULT_ID_COL):
#     """Load the survey dataset and remove trailing non-record rows with missing ID."""
#     df = pd.read_excel(
#         data_path,
#         sheet_name=sheet_name,
#         usecols="A:R",
#         engine="openpyxl",
#     )

#     if id_col not in df.columns:
#         raise KeyError(f"Expected ID column {id_col!r} was not found.")

#     df = df.loc[df[id_col].notna()].copy()
#     df = df.reset_index(drop=True)
#     return df


def normalize_code(value):
    """Normalize occupational/industry codes as display-safe strings."""
    if pd.isna(value):
        return pd.NA

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value


def concrete_prefix(code, level):
    """Return the concrete prefix at a hierarchy level, or NA if X prevents that level."""
    code = normalize_code(code)
    if pd.isna(code):
        return pd.NA

    code = str(code).strip()
    if not code:
        return pd.NA

    if len(code) < level:
        return pd.NA

    prefix = code[:level]
    if "X" in prefix.upper():
        return pd.NA

    return prefix


def add_target_levels(df, target_col=DEFAULT_TARGET_COL):
    """Add target_l1..target_l4 hierarchy columns."""
    out = df.copy()
    out[target_col] = out[target_col].map(normalize_code)

    for level in range(1, 5):
        out[f"target_l{level}"] = out[target_col].map(lambda x: concrete_prefix(x, level))

    return out


def dataframe_info_as_text(df):
    """Capture df.info() into a printable string."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def missing_values_summary(df):
    """Return missing-value counts and fractions."""
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_fraction": df.isna().mean(),
            "non_missing_count": df.notna().sum(),
            "dtype": df.dtypes.astype(str),
        }
    )
    return summary.sort_values("missing_fraction", ascending=False)


def support_distribution(series, label_name="category"):
    """Summarize support sizes across categories."""
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return pd.DataFrame()

    buckets = pd.cut(
        counts,
        bins=[0, 1, 2, 3, 5, 10, np.inf],
        labels=["1", "2", "3", "4-5", "6-10", ">10"],
        right=True,
    )

    out = (
        buckets.value_counts()
        .sort_index()
        .rename_axis("support_bucket")
        .reset_index(name="number_of_categories")
    )
    out["field"] = label_name
    return out[["field", "support_bucket", "number_of_categories"]]


def category_profile(series):
    """Return compact profile for one categorical field."""
    s = series.dropna().astype(str)
    if s.empty:
        return {
            "n_non_missing": 0,
            "n_unique": 0,
            "most_frequent_value": pd.NA,
            "most_frequent_count": 0,
            "most_frequent_fraction": 0.0,
        }

    vc = s.value_counts()
    return {
        "n_non_missing": int(s.shape[0]),
        "n_unique": int(vc.shape[0]),
        "most_frequent_value": vc.index[0],
        "most_frequent_count": int(vc.iloc[0]),
        "most_frequent_fraction": float(vc.iloc[0] / s.shape[0]),
    }


def categorical_overview(df, columns, top_n=10):
    """Return one-row summaries and top-values tables for categorical columns."""
    summaries = []
    top_values = []

    for col in columns:
        if col not in df.columns:
            continue

        profile = category_profile(df[col])
        profile["column"] = col
        summaries.append(profile)

        vc = (
            df[col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(top_n)
            .rename_axis("value")
            .reset_index(name="count")
        )
        vc["column"] = col
        vc["fraction_of_non_missing"] = vc["count"] / max(df[col].notna().sum(), 1)
        top_values.append(vc[["column", "value", "count", "fraction_of_non_missing"]])

    summary_df = pd.DataFrame(summaries)
    top_values_df = pd.concat(top_values, ignore_index=True) if top_values else pd.DataFrame()

    if not summary_df.empty:
        summary_df = summary_df[
            [
                "column",
                "n_non_missing",
                "n_unique",
                "most_frequent_value",
                "most_frequent_count",
                "most_frequent_fraction",
            ]
        ]

    return summary_df, top_values_df


def is_missing_text(value):
    """Missing-text predicate used for survey text fields."""
    if pd.isna(value) or value is None:
        return True
    return str(value).strip().lower() in STOP_WORDS


def _collect_text_values(row, text_cols=DEFAULT_TEXT_ORDER_FOR_EMBEDDINGS):
    """Collect cleaned, non-missing text values while preserving column order."""
    values = []
    for col in text_cols:
        if col in row.index:
            value = row[col]
            if not is_missing_text(value):
                cleaned = " ".join(str(value).split())
                values.append(cleaned)

    return list(dict.fromkeys(values))


def build_embedding_text(row, text_cols=DEFAULT_TEXT_ORDER_FOR_EMBEDDINGS):
    """Build the dense text used before sentence-transformer embeddings."""
    values = _collect_text_values(row, text_cols=text_cols)
    return "תיאור מקום עבודה ותפקיד: " + " ,".join(values)


def build_bow_tfidf_text(row, text_cols=DEFAULT_TEXT_ORDER_FOR_EMBEDDINGS):
    """Build plain text for BOW/TF-IDF without the sentence-transformer prefix."""
    values = _collect_text_values(row, text_cols=text_cols)
    return " | ".join(values)


def add_text_features(df, text_cols=DEFAULT_TEXT_ORDER_FOR_EMBEDDINGS):
    """Add separate text columns for embeddings and for BOW/TF-IDF analysis."""
    out = df.copy()

    out["embedding_text"] = out.apply(lambda row: build_embedding_text(row, text_cols=text_cols), axis=1)
    out["bow_tfidf_text"] = out.apply(lambda row: build_bow_tfidf_text(row, text_cols=text_cols), axis=1)

    out["text_char_count"] = out["bow_tfidf_text"].astype(str).str.len()
    out["text_word_count"] = out["bow_tfidf_text"].astype(str).str.split().map(len)

    present_count = []
    for _, row in out.iterrows():
        count = 0
        for col in text_cols:
            if col in out.columns and not is_missing_text(row.get(col)):
                count += 1
        present_count.append(count)
    out["text_non_missing_field_count"] = present_count
    return out


def bow_document_frequency(df, text_col="bow_tfidf_text", max_features=6000):
    """Calculate binary bag-of-words document frequency."""
    vectorizer = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        binary=True,
        min_df=1,
        lowercase=False,
        max_features=max_features,
    )

    X = vectorizer.fit_transform(df[text_col].astype(str).fillna(""))
    terms = np.array(vectorizer.get_feature_names_out())
    doc_freq = np.asarray(X.sum(axis=0)).ravel().astype(int)

    freq = pd.DataFrame(
        {
            "token": terms,
            "document_frequency": doc_freq,
            "document_fraction": doc_freq / max(X.shape[0], 1),
        }
    ).sort_values(["document_frequency", "token"], ascending=[False, True]).reset_index(drop=True)

    return freq, X, vectorizer


def vectorize_text(df, text_col="bow_tfidf_text", vectorizer_type="tfidf", max_features=6000):
    """Create normalized BOW or TF-IDF text vectors for exploratory analysis."""
    if vectorizer_type == "bow":
        vectorizer = CountVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            binary=True,
            min_df=1,
            lowercase=False,
            max_features=max_features,
        )
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=max_features,
        )
    else:
        raise ValueError("vectorizer_type must be 'bow' or 'tfidf'.")

    X = vectorizer.fit_transform(df[text_col].astype(str).fillna(""))
    X = normalize(X)
    return X, vectorizer


def load_embeddings_if_available(path, id_col=DEFAULT_ID_COL, embedding_col="embedding"):
    """Load embeddings if the pickle exists. Return None if unavailable."""
    path = Path(path)
    if not path.exists():
        return None

    embeddings_df = pd.read_pickle(path)

    if id_col not in embeddings_df.columns:
        raise KeyError(f"Embedding file exists but does not contain ID column {id_col!r}.")

    if embedding_col not in embeddings_df.columns:
        # Fall back to common vector-column names.
        candidates = [
            c for c in embeddings_df.columns
            if c.lower() in {"embedding", "embeddings", "vector", "sentence_embedding"}
        ]
        if candidates:
            embedding_col = candidates[0]
        else:
            raise KeyError(
                f"Embedding file exists but no vector column named {embedding_col!r} was found."
            )

    out = embeddings_df[[id_col, embedding_col]].copy()
    return out


def stack_embedding_column(merged_df, embedding_col="embedding"):
    """Stack one-column vector embeddings into a 2D numpy array."""
    valid = merged_df[embedding_col].notna()
    vectors = merged_df.loc[valid, embedding_col].map(lambda x: np.asarray(x, dtype=float)).tolist()

    if not vectors:
        return valid, np.empty((0, 0))

    matrix = np.vstack(vectors)
    matrix = normalize(matrix)
    return valid, matrix


def compute_silhouette_for_labels(X, labels, metric="cosine", min_group_size=2):
    """Compute silhouette score after dropping missing labels and very small groups."""
    labels = pd.Series(labels).astype("string")
    valid = labels.notna()

    labels_valid = labels.loc[valid]
    if hasattr(X, "iloc"):
        X_valid = X.loc[valid]
    else:
        X_valid = X[valid.to_numpy()]

    support = labels_valid.value_counts()
    keep_labels = support[support >= min_group_size].index
    keep = labels_valid.isin(keep_labels)

    labels_keep = labels_valid.loc[keep]
    if hasattr(X_valid, "iloc"):
        X_keep = X_valid.loc[keep]
    else:
        X_keep = X_valid[keep.to_numpy()]

    n_samples = int(labels_keep.shape[0])
    n_labels = int(labels_keep.nunique())

    result = {
        "n_samples_used": n_samples,
        "n_labels_used": n_labels,
        "min_group_size": min_group_size,
        "silhouette_score": np.nan,
    }

    if n_samples <= n_labels or n_labels < 2:
        return result

    try:
        result["silhouette_score"] = float(silhouette_score(X_keep, labels_keep, metric=metric))
    except Exception:
        result["silhouette_score"] = np.nan

    return result


def centroid_similarity_diagnostics(X, labels, metric_name="cosine", min_group_size=2):
    """Compare similarity to own centroid versus nearest other centroid."""
    labels = pd.Series(labels).astype("string")
    valid = labels.notna()

    if hasattr(X, "iloc"):
        X_valid = X.loc[valid]
    else:
        X_valid = X[valid.to_numpy()]

    labels_valid = labels.loc[valid].reset_index(drop=True)
    support = labels_valid.value_counts()
    keep_labels = support[support >= min_group_size].index
    keep = labels_valid.isin(keep_labels)

    labels_keep = labels_valid.loc[keep].reset_index(drop=True)

    if hasattr(X_valid, "iloc"):
        X_keep = X_valid.loc[keep]
    else:
        X_keep = X_valid[keep.to_numpy()]

    if labels_keep.nunique() < 2 or len(labels_keep) == 0:
        return pd.DataFrame()

    unique_labels = labels_keep.unique().tolist()
    centroids = []

    for label in unique_labels:
        mask = labels_keep == label
        subset = X_keep[mask.to_numpy()]
        centroid = subset.mean(axis=0)
        if hasattr(centroid, "A1"):
            centroid = centroid.A1
        centroid = np.asarray(centroid).ravel()
        centroids.append(centroid)

    centroids = normalize(np.vstack(centroids))
    similarities = cosine_similarity(X_keep, centroids)

    label_to_pos = {label: i for i, label in enumerate(unique_labels)}
    own_idx = np.array([label_to_pos[label] for label in labels_keep])
    own_sim = similarities[np.arange(similarities.shape[0]), own_idx]

    masked = similarities.copy()
    masked[np.arange(similarities.shape[0]), own_idx] = -np.inf
    nearest_other_sim = masked.max(axis=1)

    diag = pd.DataFrame(
        {
            "label": labels_keep,
            "own_centroid_similarity": own_sim,
            "nearest_other_centroid_similarity": nearest_other_sim,
            "centroid_margin": own_sim - nearest_other_sim,
        }
    )
    return diag


def summarize_centroid_margin(diag, vector_name, label_name):
    """Summarize centroid similarity diagnostics."""
    if diag is None or diag.empty:
        return {
            "vector_representation": vector_name,
            "label": label_name,
            "n_samples_used": 0,
            "mean_own_centroid_similarity": np.nan,
            "mean_nearest_other_similarity": np.nan,
            "mean_centroid_margin": np.nan,
            "fraction_positive_margin": np.nan,
        }

    return {
        "vector_representation": vector_name,
        "label": label_name,
        "n_samples_used": int(len(diag)),
        "mean_own_centroid_similarity": float(diag["own_centroid_similarity"].mean()),
        "mean_nearest_other_similarity": float(diag["nearest_other_centroid_similarity"].mean()),
        "mean_centroid_margin": float(diag["centroid_margin"].mean()),
        "fraction_positive_margin": float((diag["centroid_margin"] > 0).mean()),
    }


def pca_projection(matrix, labels=None):
    """Return a 2D PCA projection for dense vectors."""
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return pd.DataFrame()

    projection = PCA(n_components=2, random_state=42).fit_transform(matrix)
    out = pd.DataFrame({"pc1": projection[:, 0], "pc2": projection[:, 1]})
    if labels is not None:
        out["label"] = pd.Series(labels).astype("string").to_numpy()
    return out
