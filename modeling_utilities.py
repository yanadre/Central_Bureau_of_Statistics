import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize


from config import (
    ID_COL,
    TARGET_COL,
    INDUSTRY_COL,
    TEXT_COLS,
    TEXT_ORDER_FOR_BASELINE,
    SELECTED_TABULAR_FEATURES,
    SELECTED_CATEGORICAL_FEATURES,
    SELECTED_NUMERIC_FEATURES,
    SIMILARITY_FEATURES,
    MODEL_FEATURES,
    CAT_FEATURES,
    THRESHOLDS as DEFAULT_THRESHOLDS,
    MIN_SUPPORT_FOR_AUTO as DEFAULT_MIN_SUPPORT_FOR_AUTO,
    BOW_MAX_FEATURES,
    TFIDF_MAX_FEATURES,
)

# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def normalize_id_for_merge(value):
    """Normalize IDs only for matching. The original ID column is preserved."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_target(value):
    if pd.isna(value):
        return pd.NA
    return str(value).strip()


def first_x_position(code):
    """Number of concrete leading digits before first X.

    2512 -> 4
    251X -> 3
    25XX -> 2
    2XXX -> 1
    XXXX -> 0
    """
    code = str(code)
    for i, ch in enumerate(code[:4]):
        if ch.upper() == "X":
            return i
    return min(len(code), 4)


def concrete_prefix(code, level):
    """Return concrete target prefix if known, otherwise missing."""
    if pd.isna(code):
        return pd.NA
    code = str(code).strip()
    if len(code) < level:
        return pd.NA
    prefix = code[:level]
    if "X" in prefix.upper():
        return pd.NA
    return prefix


def complete_with_x(prefix, total_len=4):
    prefix = "" if prefix is None or pd.isna(prefix) else str(prefix)
    prefix = prefix[:total_len]
    return prefix + "X" * (total_len - len(prefix))


def clean_text_scalar(value):
    """Clean one text value. Missing values become empty strings."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def is_kenal(value):
    return isinstance(value, str) and value.strip() == 'כנ"ל'


def resolve_kenal(row):
    """Resolve the observed Hebrew 'same as above' token where possible."""
    row = row.copy()

    if "SugMachlaka" in row.index and is_kenal(row["SugMachlaka"]) and clean_text_scalar(row.get("SugAvoda")):
        row["SugMachlaka"] = row["SugAvoda"]

    if "TeurPeula" in row.index and is_kenal(row["TeurPeula"]) and clean_text_scalar(row.get("EzoAvoda")):
        row["TeurPeula"] = row["EzoAvoda"]

    if "TeurTafkid" in row.index and is_kenal(row["TeurTafkid"]):
        if clean_text_scalar(row.get("TeurPeula")) and not is_kenal(row.get("TeurPeula")):
            row["TeurTafkid"] = row["TeurPeula"]
        elif clean_text_scalar(row.get("EzoAvoda")):
            row["TeurTafkid"] = row["EzoAvoda"]

    return row


def build_combined_text(row):
    """Internal text input used only to create TF-IDF similarity features.

    This is not a CatBoost feature.
    """
    row = resolve_kenal(row)

    values = []
    for col in TEXT_ORDER_FOR_BASELINE:
        if col not in row.index:
            continue
        value = clean_text_scalar(row[col])
        if value == "" or value == 'כנ"ל':
            continue
        values.append(value)

    # Remove duplicate text values within the same row while preserving order.
    return " | ".join(list(dict.fromkeys(values)))


def prepare_records_from_excel(data_path, sheet_name="Survey Dataset"):
    """Load data and create minimal internal fields for modeling.

    The original ID column is preserved.
    The mistaken trailing non-record row is excluded because it has no ID and no target.
    """
    raw = pd.read_excel(
        data_path,
        sheet_name=sheet_name,
        usecols="A:R",
        dtype={TARGET_COL: "string", INDUSTRY_COL: "string"},
        engine="openpyxl",
    )

    records = raw.loc[raw[ID_COL].notna() & raw[TARGET_COL].notna()].copy()

    # Requested preprocessing for selected structured features.
    # MenahelEtMi is ordinal/numeric; missing values are coded as 0.
    # MakorSachar is categorical; missing values are interpreted as self-employed.
    # MaamadAvoda and TeudaGvoha are categorical. shnotlimud is numeric.
    # Gil is intentionally not used.
    if "MenahelEtMi" in records.columns:
        records["MenahelEtMi"] = pd.to_numeric(records["MenahelEtMi"], errors="coerce").fillna(0)

    if "MakorSachar" in records.columns:
        records["MakorSachar"] = records["MakorSachar"].where(
            records["MakorSachar"].notna(),
            "<self-employed>",
        ).astype(str)

    if "MaamadAvoda" in records.columns:
        records["MaamadAvoda"] = records["MaamadAvoda"].astype("string")

    if "TeudaGvoha" in records.columns:
        records["TeudaGvoha"] = records["TeudaGvoha"].astype("string").fillna("__MISSING__")

    if "shnotlimud" in records.columns:
        records["shnotlimud"] = pd.to_numeric(records["shnotlimud"], errors="coerce")

    records[TARGET_COL] = records[TARGET_COL].map(normalize_target)
    records["_merge_id"] = records[ID_COL].map(normalize_id_for_merge)

    records["target_depth"] = records[TARGET_COL].map(first_x_position)
    records["target_has_x"] = records["target_depth"] < 4

    for level in range(1, 5):
        records[f"target_l{level}"] = records[TARGET_COL].map(lambda x: concrete_prefix(x, level))

    records["_combined_text_for_tfidf"] = records.apply(build_combined_text, axis=1)

    # Diagnostic only, not a CatBoost feature.
    exact_counts = records["target_l4"].dropna().astype(str).value_counts()
    records["is_singleton_exact"] = records["target_l4"].astype("string").map(
        lambda x: bool(pd.notna(x) and exact_counts.get(str(x), 0) == 1)
    )

    return raw, records


# ---------------------------------------------------------------------
# Embedding loading by original ID
# ---------------------------------------------------------------------

def _read_pickle_dataframe(path):
    try:
        obj = pd.read_pickle(path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not read {path}. Missing module: {e.name}. "
            "Install the missing package or resave the pickle with standard pandas/numpy dtypes."
        ) from e

    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, dict):
        for key in ["df", "data", "records", "survey", "survey_with_embeddings"]:
            if key in obj and isinstance(obj[key], pd.DataFrame):
                return obj[key]

    raise TypeError("The pickle must contain a pandas DataFrame or a dict containing a DataFrame.")


def _parse_embedding_cell(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)

    if pd.isna(value):
        return None

    text = str(value).strip().strip("[]()")
    if not text:
        return None

    parts = [p for p in re.split(r"[,\s]+", text) if p]
    return np.asarray([float(p) for p in parts], dtype=float)


def _normalize_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_embeddings_from_pickle_by_id(
    embeddings_path,
    records,
    id_col=ID_COL,
    embedding_col=None,
    embedding_cols=None,
):
    """Load survey_with_embeddings.pkl and align embeddings to records by original ID."""
    emb_df = _read_pickle_dataframe(embeddings_path).copy()

    if id_col not in emb_df.columns:
        raise ValueError(f"Embedding pickle must contain the original ID column: {id_col}")

    emb_df["_merge_id"] = emb_df[id_col].map(normalize_id_for_merge)
    emb_df = emb_df.loc[emb_df["_merge_id"].notna()].copy()

    if emb_df["_merge_id"].duplicated().any():
        dup_ids = emb_df.loc[emb_df["_merge_id"].duplicated(), id_col].head(10).tolist()
        raise ValueError(f"Embedding pickle contains duplicated IDs, for example: {dup_ids}")

    align = records[[id_col, "_merge_id"]].copy()

    if embedding_col is not None:
        if embedding_col not in emb_df.columns:
            raise ValueError(f"embedding_col='{embedding_col}' was not found in the pickle.")

        tmp = align.merge(
            emb_df[["_merge_id", embedding_col]],
            on="_merge_id",
            how="left",
            validate="one_to_one",
        )

        if tmp[embedding_col].isna().any():
            missing = tmp.loc[tmp[embedding_col].isna(), id_col].head(10).tolist()
            raise ValueError(f"Missing embeddings for some records, for example IDs: {missing}")

        parsed = tmp[embedding_col].map(_parse_embedding_cell)
        return _normalize_matrix(np.vstack(parsed.to_numpy()))

    if embedding_cols is None:
        prefix_cols = [
            c for c in emb_df.columns
            if str(c).lower().startswith(("emb_", "embedding_", "dim_", "vector_"))
            and pd.api.types.is_numeric_dtype(emb_df[c])
        ]

        if prefix_cols:
            embedding_cols = prefix_cols
        else:
            possible_single_cols = [
                c for c in emb_df.columns
                if ("emb" in str(c).lower() or "vector" in str(c).lower())
                and c not in [id_col, "_merge_id"]
            ]
            if len(possible_single_cols) == 1:
                return load_embeddings_from_pickle_by_id(
                    embeddings_path,
                    records,
                    id_col=id_col,
                    embedding_col=possible_single_cols[0],
                    embedding_cols=None,
                )

            non_embedding = {
                id_col, "_merge_id", TARGET_COL, INDUSTRY_COL,
                *TEXT_COLS, *SELECTED_TABULAR_FEATURES,
            }
            embedding_cols = [
                c for c in emb_df.columns
                if c not in non_embedding and pd.api.types.is_numeric_dtype(emb_df[c])
            ]

    if not embedding_cols:
        raise ValueError(
            "Could not infer embedding columns. "
            "Pass embedding_col='...' for a single vector column or embedding_cols=[...] for wide format."
        )

    tmp = align.merge(
        emb_df[["_merge_id"] + list(embedding_cols)],
        on="_merge_id",
        how="left",
        validate="one_to_one",
    )

    if tmp[list(embedding_cols)].isna().any().any():
        missing = tmp.loc[tmp[list(embedding_cols)].isna().any(axis=1), id_col].head(10).tolist()
        raise ValueError(f"Missing embeddings for some records, for example IDs: {missing}")

    return _normalize_matrix(tmp[list(embedding_cols)].to_numpy(dtype=float))


# ---------------------------------------------------------------------
# Centroid similarity features
# ---------------------------------------------------------------------

def cosine_to_centroid(sample_vector, cluster_vectors):
    """Cosine similarity to a centroid.

    Supports dense sentence-embedding arrays and sparse TF-IDF matrices.
    """
    if cluster_vectors.shape[0] == 0:
        return np.nan, 0, np.nan

    # Sparse TF-IDF path.
    if hasattr(cluster_vectors, "toarray"):
        centroid = np.asarray(cluster_vectors.mean(axis=0)).ravel()
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return np.nan, cluster_vectors.shape[0], np.nan

        centroid = centroid / norm
        sim_raw = sample_vector.dot(centroid)
        sim = float(sim_raw[0]) if hasattr(sim_raw, "__len__") else float(sim_raw)

        if cluster_vectors.shape[0] >= 2:
            member_sims = cluster_vectors @ centroid
            sparsity = float(1.0 - np.mean(member_sims))
        else:
            sparsity = np.nan

        return sim, int(cluster_vectors.shape[0]), sparsity

    # Dense sentence-embedding path.
    centroid = cluster_vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return np.nan, len(cluster_vectors), np.nan

    centroid = centroid / norm
    sim = float(np.dot(sample_vector, centroid))

    if len(cluster_vectors) >= 2:
        member_sims = cluster_vectors @ centroid
        sparsity = float(1.0 - np.mean(member_sims))
    else:
        sparsity = np.nan

    return sim, int(len(cluster_vectors)), sparsity


def candidate_prefixes(train_records, level, current_prefix=""):
    col = f"target_l{level}"
    values = train_records[col].dropna().astype(str).unique().tolist()
    return sorted([v for v in values if v.startswith(current_prefix)])


def add_requested_tabular_values(row_dict, source_row):
    """Add exactly the requested structured values."""
    for col in SELECTED_TABULAR_FEATURES:
        value = source_row[col] if col in source_row.index else np.nan

        if col == "MakorSachar" and pd.isna(value):
            row_dict[col] = "<self-employed>"
        elif col in SELECTED_CATEGORICAL_FEATURES:
            row_dict[col] = "__MISSING__" if pd.isna(value) else str(value)
        elif col == "MenahelEtMi":
            row_dict[col] = 0 if pd.isna(value) else value
        else:
            row_dict[col] = value
    return row_dict


def build_candidate_feature_row(
    source_records,
    source_vectors,
    sample_row,
    sample_vector,
    candidate,
    level,
    exclude_index=None,
):
    """Build candidate-row features.

    CatBoost features remain strictly sim_l1..sim_l4 plus the six requested fields.
    Support/sparsity are metadata only.

    For training rows, exclude_index is the current observation. This prevents the
    current sample from being included in its own centroid.
    """
    row = {
        "candidate": str(candidate),
        "candidate_level": level,
    }

    for j in range(1, 5):
        sim_col = f"sim_l{j}"
        support_col = f"support_l{j}"
        sparsity_col = f"sparsity_l{j}"

        if j <= level:
            prefix_j = str(candidate)[:j]
            mask = source_records[f"target_l{j}"].astype("string") == prefix_j

            if exclude_index is not None:
                mask = mask & (source_records.index != exclude_index)

            mask_array = mask.fillna(False).to_numpy(dtype=bool)
            cluster_vectors = source_vectors[mask_array]

            sim, support, sparsity = cosine_to_centroid(sample_vector, cluster_vectors)
            row[sim_col] = sim
            row[support_col] = support
            row[sparsity_col] = sparsity
        else:
            row[sim_col] = np.nan
            row[support_col] = np.nan
            row[sparsity_col] = np.nan

    row = add_requested_tabular_values(row, sample_row)
    return row


def build_training_candidate_table(train_records, train_vectors, level):
    """Build binary row-candidate data for one hierarchy level.

    Current row is always excluded from centroid calculation.
    If the true positive candidate has zero support after exclusion, that row is
    skipped for this level. This is the correct behavior for singletons: they can
    still contribute to other rows' centroids, but they cannot train on a centroid
    made only from themselves.
    """
    target_col = f"target_l{level}"
    eligible = train_records.loc[train_records[target_col].notna()].copy()
    prefixes = candidate_prefixes(train_records, level)

    rows = []
    skipped_positive_no_centroid = 0

    for idx, sample_row in eligible.iterrows():
        local_pos = train_records.index.get_loc(idx)
        sample_vector = train_vectors[local_pos]
        true_prefix = str(sample_row[target_col])

        sample_rows = []
        positive_available = False

        for cand in prefixes:
            feat = build_candidate_feature_row(
                source_records=train_records,
                source_vectors=train_vectors,
                sample_row=sample_row,
                sample_vector=sample_vector,
                candidate=cand,
                level=level,
                exclude_index=idx,
            )

            # No centroid exists for this candidate after excluding current row.
            if feat[f"support_l{level}"] == 0:
                continue

            feat["row_index"] = idx
            feat["is_correct"] = int(str(cand) == true_prefix)

            if feat["is_correct"] == 1:
                positive_available = True

            sample_rows.append(feat)

        if positive_available:
            rows.extend(sample_rows)
        else:
            skipped_positive_no_centroid += 1

    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["skipped_positive_no_centroid"] = skipped_positive_no_centroid
    return out


def build_prediction_candidate_table(
    train_records,
    train_vectors,
    valid_row,
    valid_vector,
    level,
    current_prefix,
):
    """Build candidate rows for a validation observation.

    Validation rows are not in train_records, so no current-row exclusion is needed.
    """
    prefixes = candidate_prefixes(train_records, level, current_prefix=current_prefix)
    rows = []

    for cand in prefixes:
        feat = build_candidate_feature_row(
            source_records=train_records,
            source_vectors=train_vectors,
            sample_row=valid_row,
            sample_vector=valid_vector,
            candidate=cand,
            level=level,
            exclude_index=None,
        )

        if feat[f"support_l{level}"] == 0:
            continue

        rows.append(feat)

    return pd.DataFrame(rows)


def prepare_model_frame(candidate_df):
    """Return CatBoost input with exactly MODEL_FEATURES."""
    X = candidate_df[MODEL_FEATURES].copy()

    for col in SELECTED_CATEGORICAL_FEATURES:
        if col == "MakorSachar":
            X[col] = X[col].astype("string").fillna("<self-employed>").astype(str)
        else:
            X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)

    for col in SELECTED_NUMERIC_FEATURES + SIMILARITY_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if "MenahelEtMi" in X.columns:
        X["MenahelEtMi"] = X["MenahelEtMi"].fillna(0)

    return X


# ---------------------------------------------------------------------
# CatBoost models
# ---------------------------------------------------------------------

class ConstantProbabilityModel:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, X):
        p = np.repeat(self.probability, len(X))
        return np.vstack([1 - p, p]).T


def fit_level_model(candidate_df, catboost_params=None):
    if candidate_df.empty:
        return ConstantProbabilityModel(0.0)

    y = candidate_df["is_correct"].astype(int)

    if y.nunique() == 1:
        return ConstantProbabilityModel(float(y.iloc[0]))

    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("catboost is required. Install it with: pip install catboost") from e

    params = {
        "iterations": 80,
        "depth": 3,
        "learning_rate": 0.08,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "auto_class_weights": "Balanced",
    }
    if catboost_params:
        params.update(catboost_params)

    X = prepare_model_frame(candidate_df)

    model = CatBoostClassifier(**params)
    cat_feature_indices = [X.columns.get_loc(c) for c in CAT_FEATURES if c in X.columns]
    model.fit(X, y, cat_features=cat_feature_indices)
    return model


def fit_fold_models(train_records, train_vectors, catboost_params=None):
    level_models = {}
    for level in range(1, 5):
        cand_df = build_training_candidate_table(train_records, train_vectors, level)
        level_models[level] = fit_level_model(cand_df, catboost_params=catboost_params)
    return level_models


# ---------------------------------------------------------------------
# Beam-search prediction
# ---------------------------------------------------------------------

def _beam_item(prefix, path_score, probabilities):
    return {
        "prefix": str(prefix),
        "path_score": float(path_score),
        "probabilities": list(probabilities),
    }


def predict_fold(
    train_records,
    train_vectors,
    valid_records,
    valid_vectors,
    level_models,
    thresholds=None,
    min_support_for_auto=None,
    min_probability_margin=0.03,
    beam_width=3,
    max_children_per_parent=5,
):
    """Predict with bounded beam search.

    Keeps multiple paths alive, so an early second-best branch can recover later.
    Final prediction is still the deepest confident prefix plus X suffixes.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    min_support_for_auto = min_support_for_auto or DEFAULT_MIN_SUPPORT_FOR_AUTO

    outputs = []

    for local_pos, (idx, valid_row) in enumerate(valid_records.iterrows()):
        valid_vector = valid_vectors[local_pos]

        beam = [_beam_item(prefix="", path_score=1.0, probabilities=[])]
        accepted_prefix = ""
        accepted_probabilities = []

        out = {
            "row_index": idx,
            "predicted_code": None,
            "predicted_depth": 0,
            "model_confidence": 0.0,
            "rejected_level_confidence": np.nan,
            "stop_reason": None,
        }

        stopped_early = False

        for level in range(1, 5):
            all_candidates = []

            for parent in beam:
                candidates = build_prediction_candidate_table(
                    train_records=train_records,
                    train_vectors=train_vectors,
                    valid_row=valid_row,
                    valid_vector=valid_vector,
                    level=level,
                    current_prefix=parent["prefix"],
                )

                if candidates.empty:
                    continue

                X = prepare_model_frame(candidates)
                probs = level_models[level].predict_proba(X)[:, 1]

                candidates = candidates.copy()
                candidates["model_probability"] = probs
                candidates = candidates.sort_values("model_probability", ascending=False)

                if max_children_per_parent is not None:
                    candidates = candidates.head(max_children_per_parent)

                for _, cand in candidates.iterrows():
                    probability = float(cand["model_probability"])
                    path_score = float(parent["path_score"] * max(probability, 1e-9))
                    probabilities = parent["probabilities"] + [probability]

                    all_candidates.append({
                        "candidate": str(cand["candidate"]),
                        "level": level,
                        "model_probability": probability,
                        "path_score": path_score,
                        "probabilities": probabilities,
                        "row": cand,
                    })

            if not all_candidates:
                out["stop_reason"] = f"no_candidate_level_{level}"
                stopped_early = True
                break

            all_candidates = sorted(all_candidates, key=lambda x: x["path_score"], reverse=True)
            top = all_candidates[0]
            second_path_score = all_candidates[1]["path_score"] if len(all_candidates) > 1 else 0.0
            path_margin = top["path_score"] - second_path_score

            top_row = top["row"]
            top_probability = top["model_probability"]
            top_support = int(top_row[f"support_l{level}"]) if pd.notna(top_row[f"support_l{level}"]) else 0

            out[f"level_{level}_candidate"] = str(top["candidate"])
            out[f"level_{level}_probability"] = float(top_probability)
            out[f"level_{level}_path_score"] = float(top["path_score"])
            out[f"level_{level}_probability_margin"] = float(path_margin)
            out[f"level_{level}_similarity"] = float(top_row[f"sim_l{level}"]) if pd.notna(top_row[f"sim_l{level}"]) else np.nan
            out[f"level_{level}_support"] = top_support
            out[f"level_{level}_sparsity"] = float(top_row[f"sparsity_l{level}"]) if pd.notna(top_row[f"sparsity_l{level}"]) else np.nan

            level_accepted = (
                top_probability >= thresholds[level]
                and path_margin >= min_probability_margin
                and top_support >= min_support_for_auto[level]
            )

            out[f"level_{level}_accepted"] = bool(level_accepted)

            if level_accepted:
                accepted_prefix = str(top["candidate"])
                accepted_probabilities = top["probabilities"]
            else:
                out["rejected_level_confidence"] = float(top_probability)

                if top_probability < thresholds[level]:
                    out[f"level_{level}_rejection_reason"] = f"low_probability_level_{level}"
                elif path_margin < min_probability_margin:
                    out[f"level_{level}_rejection_reason"] = f"small_probability_margin_level_{level}"
                elif top_support < min_support_for_auto[level]:
                    out[f"level_{level}_rejection_reason"] = f"low_support_level_{level}"

            beam = [
                _beam_item(
                    prefix=item["candidate"],
                    path_score=item["path_score"],
                    probabilities=item["probabilities"],
                )
                for item in all_candidates[:beam_width]
            ]

        predicted_depth = len(accepted_prefix)

        if predicted_depth == 4:
            out["stop_reason"] = "full_prediction"
        elif predicted_depth > 0:
            out["stop_reason"] = f"beam_uncertain_after_level_{predicted_depth}"
        elif stopped_early and out["stop_reason"] is not None:
            pass
        else:
            out["stop_reason"] = "beam_no_confident_prefix"

        out["predicted_code"] = complete_with_x(accepted_prefix)
        out["predicted_depth"] = first_x_position(out["predicted_code"])
        out["model_confidence"] = float(min(accepted_probabilities)) if accepted_probabilities else 0.0
        out["beam_width"] = beam_width
        out["max_children_per_parent"] = max_children_per_parent

        outputs.append(out)

    return pd.DataFrame(outputs).set_index("row_index")



# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------

def extract_feature_importance(level_models, model_name, fold):
    """Extract CatBoost feature importance for each hierarchy level.

    CatBoost is trained separately per hierarchy level, so importance is reported
    per fold and per level. Constant fallback models receive zero importance.
    """
    rows = []

    for level, model in level_models.items():
        if hasattr(model, "get_feature_importance"):
            try:
                importances = np.asarray(model.get_feature_importance(), dtype=float)
            except Exception:
                importances = np.zeros(len(MODEL_FEATURES), dtype=float)
        else:
            importances = np.zeros(len(MODEL_FEATURES), dtype=float)

        if len(importances) != len(MODEL_FEATURES):
            fixed = np.zeros(len(MODEL_FEATURES), dtype=float)
            fixed[: min(len(importances), len(fixed))] = importances[: min(len(importances), len(fixed))]
            importances = fixed

        total = float(np.nansum(importances))
        normalized = importances / total if total > 0 else np.zeros_like(importances)

        for feature, importance, importance_normalized in zip(MODEL_FEATURES, importances, normalized):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "level": level,
                    "feature": feature,
                    "importance": float(importance),
                    "importance_normalized": float(importance_normalized),
                }
            )

    return pd.DataFrame(rows)


def summarize_feature_importance(feature_importance):
    """Average feature importance across folds and hierarchy levels."""
    if feature_importance is None or len(feature_importance) == 0:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "importance_normalized_mean"])

    return (
        feature_importance
        .groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_normalized_mean=("importance_normalized", "mean"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def summarize_feature_importance_by_level(feature_importance):
    """Average feature importance across folds, separately for each hierarchy level."""
    if feature_importance is None or len(feature_importance) == 0:
        return pd.DataFrame(columns=["level", "feature", "importance_mean", "importance_std", "importance_normalized_mean"])

    return (
        feature_importance
        .groupby(["level", "feature"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_normalized_mean=("importance_normalized", "mean"),
        )
        .sort_values(["level", "importance_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------
# CV runners
# ---------------------------------------------------------------------

def make_kfold_splits(records, n_splits=5, random_state=42):
    """Regular KFold splits. No stratification and no grouping."""
    effective_splits = min(n_splits, len(records))
    if effective_splits < 2:
        raise ValueError("Need at least two records for KFold.")

    splitter = KFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = list(splitter.split(records))

    split_info = {
        "cv_type": "KFold",
        "effective_n_splits": effective_splits,
    }

    return splits, split_info


def pattern_safe_match(pred, true):
    pred = str(pred)
    true = str(true)

    pred_depth = first_x_position(pred)
    true_depth = first_x_position(true)

    compare_len = min(pred_depth, true_depth)

    if pred[:compare_len] != true[:compare_len]:
        return False

    if pred_depth > true_depth:
        return False

    return True


def hierarchical_prefix_score(pred, true):
    pred = str(pred)
    true = str(true)
    true_depth = first_x_position(true)

    if true_depth == 0:
        return 1.0 if first_x_position(pred) == 0 else 0.0

    score = 0
    for i in range(true_depth):
        if i >= len(pred) or pred[i].upper() == "X":
            break
        if pred[i] == true[i]:
            score += 1
        else:
            break

    return score / true_depth


def attach_training_support_diagnostics(predictions, train_records_by_fold):
    rows = []

    for idx, row in predictions.iterrows():
        fold = row["fold"]
        train_records = train_records_by_fold[fold]
        true_code = str(row[TARGET_COL])

        diag = {"row_index": idx}
        for level in range(1, 5):
            true_prefix = concrete_prefix(true_code, level)
            if pd.isna(true_prefix):
                diag[f"true_support_l{level}"] = np.nan
            else:
                diag[f"true_support_l{level}"] = int(
                    (train_records[f"target_l{level}"].astype("string") == str(true_prefix)).fillna(False).sum()
                )
        rows.append(diag)

    diag_df = pd.DataFrame(rows).set_index("row_index")
    return predictions.join(diag_df)


def finalize_predictions(predictions, records, split_info):
    predictions = predictions.sort_index()

    expected = set(records.index)
    actual = set(predictions.index)
    if expected != actual:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise RuntimeError(f"OOF prediction coverage mismatch. Missing={missing}; Extra={extra}")

    predictions["needs_manual_tagging"] = predictions["predicted_depth"] < 4
    predictions["is_safe_prediction"] = [
        pattern_safe_match(pred, true)
        for pred, true in zip(predictions["predicted_code"], predictions[TARGET_COL])
    ]
    predictions["hierarchical_prefix_score"] = [
        hierarchical_prefix_score(pred, true)
        for pred, true in zip(predictions["predicted_code"], predictions[TARGET_COL])
    ]

    predictions["is_singleton_exact"] = records.loc[predictions.index, "is_singleton_exact"].astype(bool).values
    predictions["cv_type"] = split_info["cv_type"]
    predictions["effective_n_splits"] = split_info["effective_n_splits"]

    return predictions


def run_cv_with_vectors(
    records,
    vectors,
    model_name,
    n_splits=5,
    thresholds=None,
    min_support_for_auto=None,
    min_probability_margin=0.03,
    beam_width=3,
    max_children_per_parent=5,
    catboost_params=None,
    random_state=42,
    return_feature_importance=False,
):
    records = records.copy()
    vectors = vectors if hasattr(vectors, "toarray") else np.asarray(vectors)

    splits, split_info = make_kfold_splits(
        records=records,
        n_splits=n_splits,
        random_state=random_state,
    )

    all_predictions = []
    feature_importance_frames = []
    train_records_by_fold = {}

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_records = records.iloc[train_idx].copy()
        valid_records = records.iloc[valid_idx].copy()

        train_vectors = vectors[train_idx]
        valid_vectors = vectors[valid_idx]

        level_models = fit_fold_models(train_records, train_vectors, catboost_params=catboost_params)

        if return_feature_importance:
            feature_importance_frames.append(
                extract_feature_importance(level_models, model_name=model_name, fold=fold)
            )

        fold_pred = predict_fold(
            train_records=train_records,
            train_vectors=train_vectors,
            valid_records=valid_records,
            valid_vectors=valid_vectors,
            level_models=level_models,
            thresholds=thresholds,
            min_support_for_auto=min_support_for_auto,
            min_probability_margin=min_probability_margin,
            beam_width=beam_width,
            max_children_per_parent=max_children_per_parent,
        )

        fold_out = valid_records[[ID_COL, TARGET_COL]].copy().join(fold_pred)
        fold_out["fold"] = fold
        fold_out["model"] = model_name
        all_predictions.append(fold_out)

        train_records_by_fold[fold] = train_records

    predictions = pd.concat(all_predictions).sort_index()
    predictions = finalize_predictions(predictions, records, split_info)
    predictions = attach_training_support_diagnostics(predictions, train_records_by_fold)

    if return_feature_importance:
        feature_importance = (
            pd.concat(feature_importance_frames, ignore_index=True)
            if feature_importance_frames
            else pd.DataFrame()
        )
        return predictions, feature_importance

    return predictions

def run_tfidf_baseline_cv(
    records,
    n_splits=5,
    thresholds=None,
    min_support_for_auto=None,
    min_probability_margin=0.03,
    beam_width=3,
    max_children_per_parent=5,
    catboost_params=None,
    random_state=42,
    return_feature_importance=False,
):
    """TF-IDF baseline.

    TF-IDF is used only to build sim_l1..sim_l4 centroid similarity features.
    CatBoost receives only MODEL_FEATURES.
    """
    splits, split_info = make_kfold_splits(
        records=records,
        n_splits=n_splits,
        random_state=random_state,
    )

    all_predictions = []
    feature_importance_frames = []
    train_records_by_fold = {}

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_records = records.iloc[train_idx].copy()
        valid_records = records.iloc[valid_idx].copy()

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=TFIDF_MAX_FEATURES,
        )

        train_vectors = vectorizer.fit_transform(train_records["_combined_text_for_tfidf"].fillna("").astype(str))
        valid_vectors = vectorizer.transform(valid_records["_combined_text_for_tfidf"].fillna("").astype(str))

        train_vectors = normalize(train_vectors)
        valid_vectors = normalize(valid_vectors)

        level_models = fit_fold_models(train_records, train_vectors, catboost_params=catboost_params)

        if return_feature_importance:
            feature_importance_frames.append(
                extract_feature_importance(
                    level_models,
                    model_name="baseline_tfidf_similarity_catboost",
                    fold=fold,
                )
            )

        fold_pred = predict_fold(
            train_records=train_records,
            train_vectors=train_vectors,
            valid_records=valid_records,
            valid_vectors=valid_vectors,
            level_models=level_models,
            thresholds=thresholds,
            min_support_for_auto=min_support_for_auto,
            min_probability_margin=min_probability_margin,
            beam_width=beam_width,
            max_children_per_parent=max_children_per_parent,
        )

        fold_out = valid_records[[ID_COL, TARGET_COL]].copy().join(fold_pred)
        fold_out["fold"] = fold
        fold_out["model"] = "baseline_tfidf_similarity_catboost"
        all_predictions.append(fold_out)

        train_records_by_fold[fold] = train_records

    predictions = pd.concat(all_predictions).sort_index()
    predictions = finalize_predictions(predictions, records, split_info)
    predictions = attach_training_support_diagnostics(predictions, train_records_by_fold)

    if return_feature_importance:
        feature_importance = (
            pd.concat(feature_importance_frames, ignore_index=True)
            if feature_importance_frames
            else pd.DataFrame()
        )
        return predictions, feature_importance

    return predictions


def run_bow_baseline_cv(
    records,
    n_splits=5,
    thresholds=None,
    min_support_for_auto=None,
    min_probability_margin=0.03,
    beam_width=3,
    max_children_per_parent=5,
    catboost_params=None,
    random_state=42,
    return_feature_importance=False,
):
    """Binary Bag-of-Words baseline.

    Each word is counted once per record using CountVectorizer(binary=True).
    BOW vectors are used only to build sim_l1..sim_l4 centroid similarity features.
    CatBoost receives only MODEL_FEATURES.
    """
    splits, split_info = make_kfold_splits(
        records=records,
        n_splits=n_splits,
        random_state=random_state,
    )

    all_predictions = []
    feature_importance_frames = []
    train_records_by_fold = {}

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_records = records.iloc[train_idx].copy()
        valid_records = records.iloc[valid_idx].copy()

        vectorizer = CountVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            binary=True,
            min_df=1,
            lowercase=False,
            max_features=BOW_MAX_FEATURES,
        )

        train_vectors = vectorizer.fit_transform(train_records["_combined_text_for_tfidf"].fillna("").astype(str))
        valid_vectors = vectorizer.transform(valid_records["_combined_text_for_tfidf"].fillna("").astype(str))

        train_vectors = normalize(train_vectors)
        valid_vectors = normalize(valid_vectors)

        level_models = fit_fold_models(train_records, train_vectors, catboost_params=catboost_params)

        if return_feature_importance:
            feature_importance_frames.append(
                extract_feature_importance(
                    level_models,
                    model_name="baseline_bow_similarity_catboost",
                    fold=fold,
                )
            )

        fold_pred = predict_fold(
            train_records=train_records,
            train_vectors=train_vectors,
            valid_records=valid_records,
            valid_vectors=valid_vectors,
            level_models=level_models,
            thresholds=thresholds,
            min_support_for_auto=min_support_for_auto,
            min_probability_margin=min_probability_margin,
            beam_width=beam_width,
            max_children_per_parent=max_children_per_parent,
        )

        fold_out = valid_records[[ID_COL, TARGET_COL]].copy().join(fold_pred)
        fold_out["fold"] = fold
        fold_out["model"] = "baseline_bow_similarity_catboost"
        all_predictions.append(fold_out)

        train_records_by_fold[fold] = train_records

    predictions = pd.concat(all_predictions).sort_index()
    predictions = finalize_predictions(predictions, records, split_info)
    predictions = attach_training_support_diagnostics(predictions, train_records_by_fold)

    if return_feature_importance:
        feature_importance = (
            pd.concat(feature_importance_frames, ignore_index=True)
            if feature_importance_frames
            else pd.DataFrame()
        )
        return predictions, feature_importance

    return predictions

def run_embedding_main_cv(
    records,
    embeddings,
    n_splits=5,
    thresholds=None,
    min_support_for_auto=None,
    min_probability_margin=0.03,
    beam_width=3,
    max_children_per_parent=5,
    catboost_params=None,
    random_state=42,
    return_feature_importance=False,
):
    """Main model.

    Supplied sentence embeddings are used only to build sim_l1..sim_l4 centroid
    similarity features. CatBoost receives only MODEL_FEATURES.
    """
    return run_cv_with_vectors(
        records=records,
        vectors=embeddings,
        model_name="main_embedding_similarity_catboost",
        n_splits=n_splits,
        thresholds=thresholds,
        min_support_for_auto=min_support_for_auto,
        min_probability_margin=min_probability_margin,
        beam_width=beam_width,
        max_children_per_parent=max_children_per_parent,
        catboost_params=catboost_params,
        random_state=random_state,
        return_feature_importance=return_feature_importance,
    )


# ---------------------------------------------------------------------
# Global metrics and interpretation
# ---------------------------------------------------------------------

def evaluate_global_predictions(predictions):
    y_true = predictions[TARGET_COL].astype(str)
    y_pred = predictions["predicted_code"].astype(str)

    true_depth = y_true.map(first_x_position)
    pred_depth = y_pred.map(first_x_position)

    manual = pred_depth < 4
    auto = pred_depth == 4
    concrete_true = true_depth == 4
    x_true = true_depth < 4
    singleton = predictions["is_singleton_exact"].astype(bool).to_numpy()
    non_singleton = ~singleton

    safe_flags = predictions["is_safe_prediction"].astype(bool).to_numpy()

    metrics = {
        "n_samples": len(predictions),
        "cv_type": predictions["cv_type"].iloc[0],
        "effective_n_splits": int(predictions["effective_n_splits"].iloc[0]),
        "singleton_exact_n": int(singleton.sum()),
        "singleton_exact_fraction": float(singleton.mean()),
        "non_singleton_n": int(non_singleton.sum()),
        "manual_tagging_n": int(manual.sum()),
        "manual_tagging_fraction": float(manual.mean()),
        "auto_tagging_n": int(auto.sum()),
        "auto_tagging_fraction": float(auto.mean()),
        "safe_partial_accuracy_all": float(safe_flags.mean()),
        "hierarchical_prefix_score_all": float(predictions["hierarchical_prefix_score"].mean()),
        "mean_predicted_digits": float(pred_depth.mean()),
        "full_code_prediction_rate": float(auto.mean()),
        "manual_tagging_fraction_non_singleton": float(manual[non_singleton].mean()) if non_singleton.any() else np.nan,
        "manual_tagging_fraction_singleton_exact": float(manual[singleton].mean()) if singleton.any() else np.nan,
        "safe_partial_accuracy_non_singleton": float(safe_flags[non_singleton].mean()) if non_singleton.any() else np.nan,
        "safe_partial_accuracy_singleton_exact": float(safe_flags[singleton].mean()) if singleton.any() else np.nan,
    }

    if auto.any():
        metrics["accuracy_out_of_auto_tagged_samples"] = float(safe_flags[auto].mean())
        metrics["error_rate_out_of_auto_tagged_samples"] = float(1.0 - safe_flags[auto].mean())
        metrics["mean_confidence_auto_tagged"] = float(predictions.loc[auto, "model_confidence"].mean())
    else:
        metrics["accuracy_out_of_auto_tagged_samples"] = np.nan
        metrics["error_rate_out_of_auto_tagged_samples"] = np.nan
        metrics["mean_confidence_auto_tagged"] = np.nan

    if (auto & non_singleton).any():
        metrics["accuracy_out_of_auto_tagged_non_singleton"] = float(safe_flags[auto & non_singleton].mean())
    else:
        metrics["accuracy_out_of_auto_tagged_non_singleton"] = np.nan

    if (auto & singleton).any():
        metrics["accuracy_out_of_auto_tagged_singleton_exact"] = float(safe_flags[auto & singleton].mean())
    else:
        metrics["accuracy_out_of_auto_tagged_singleton_exact"] = np.nan

    if manual.any():
        metrics["manual_safe_accuracy"] = float(safe_flags[manual].mean())
        metrics["mean_confidence_manual_tagged"] = float(predictions.loc[manual, "model_confidence"].mean())
    else:
        metrics["manual_safe_accuracy"] = np.nan
        metrics["mean_confidence_manual_tagged"] = np.nan

    if (auto & concrete_true).any():
        metrics["auto_exact_accuracy_on_concrete_true"] = float(
            accuracy_score(y_true[auto & concrete_true], y_pred[auto & concrete_true])
        )
    else:
        metrics["auto_exact_accuracy_on_concrete_true"] = np.nan

    if x_true.any():
        metrics["x_target_safety_rate"] = float(safe_flags[x_true].mean())
        metrics["x_target_overspecific_rate"] = float(np.mean(pred_depth[x_true].to_numpy() > true_depth[x_true].to_numpy()))
    else:
        metrics["x_target_safety_rate"] = np.nan
        metrics["x_target_overspecific_rate"] = np.nan

    for level in range(1, 5):
        eligible = true_depth >= level
        predicted = pred_depth >= level
        metrics[f"level_{level}_coverage"] = float(predicted.mean())

        if (eligible & predicted).any():
            metrics[f"level_{level}_accuracy_when_predicted"] = float(
                np.mean([p[:level] == t[:level] for p, t in zip(y_pred[eligible & predicted], y_true[eligible & predicted])])
            )
        else:
            metrics[f"level_{level}_accuracy_when_predicted"] = np.nan

    return pd.DataFrame([metrics]).T.rename(columns={0: "value"})


def compare_models(prediction_dict):
    rows = []
    for name, pred in prediction_dict.items():
        metrics = evaluate_global_predictions(pred)["value"].to_dict()
        metrics["model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("model").round(3)



def summarize_operational_metrics(prediction_dict):
    """Compact table focused on the business decision: auto-code or manual-tag."""
    rows = []
    for model_name, pred in prediction_dict.items():
        pred = pred.copy()
        non_manual = ~pred["needs_manual_tagging"].astype(bool)
        manual = pred["needs_manual_tagging"].astype(bool)
        safe = pred["is_safe_prediction"].astype(bool)

        rows.append(
            {
                "model": model_name,
                "n_samples": int(len(pred)),
                "manual_tagging_n": int(manual.sum()),
                "manual_tagging_fraction": float(manual.mean()),
                "non_manual_n": int(non_manual.sum()),
                "non_manual_fraction": float(non_manual.mean()),
                "accuracy_among_non_manual": float(safe[non_manual].mean()) if non_manual.any() else np.nan,
                "errors_among_non_manual_n": int((non_manual & ~safe).sum()),
                "mean_confidence_non_manual": float(pred.loc[non_manual, "model_confidence"].mean()) if non_manual.any() else np.nan,
                "safe_partial_accuracy_all": float(safe.mean()),
            }
        )

    return pd.DataFrame(rows).set_index("model")



def model_context_columns(records):
    """Columns used to inspect rows in error analysis.

    The model uses similarity features internally, but for human inspection we show
    the original text fields instead of sim_l1..sim_l4.
    """
    preferred = [
        ID_COL,
        TARGET_COL,
        "target_l1",
        "target_l2",
        "target_l3",
        "target_l4",
    ]
    preferred += [c for c in TEXT_COLS if c not in preferred]
    preferred += [c for c in SELECTED_TABULAR_FEATURES if c not in preferred]
    return [c for c in preferred if c in records.columns]


def row_context(records, indices):
    """Return full context rows for display, using text columns instead of similarity features."""
    cols = model_context_columns(records)
    if len(indices) == 0:
        return pd.DataFrame(columns=cols)
    return records.loc[list(indices), cols].copy()


def get_misclassification_class_comparisons(
    predictions,
    records,
    model_name=None,
    n_cases=5,
    max_rows_per_group=None,
    random_state=42,
):
    """Return sampled non-singleton, non-manual mistakes with class-level context.

    For each sampled mistake, returns a dictionary with:
    - case_row: the misclassified record, including prediction metadata;
    - predicted_wrong_class_rows: rows from the predicted class/prefix;
    - true_right_class_rows: rows from the true exact class.

    Display columns intentionally show the original text fields plus the selected
    tabular fields. Similarity features sim_l1..sim_l4 are not shown because they
    are derived features, not human-readable raw inputs.
    """
    pred = predictions.copy()
    mask = (
        (~pred["needs_manual_tagging"].astype(bool))
        & (~pred["is_safe_prediction"].astype(bool))
        & (~pred["is_singleton_exact"].astype(bool))
        & (~pred[TARGET_COL].astype(str).str.contains("X", case=False, na=False))
    )

    cases = pred.loc[mask].copy()
    if cases.empty:
        return []

    cases = cases.sample(n=min(n_cases, len(cases)), random_state=random_state)
    outputs = []

    for case_number, (idx, case) in enumerate(cases.iterrows(), start=1):
        true_code = str(case[TARGET_COL])
        predicted_code = str(case["predicted_code"])
        predicted_depth = int(case.get("predicted_depth", 0) or 0)
        predicted_prefix = predicted_code.replace("X", "")[:predicted_depth]

        case_row = row_context(records, [idx])
        case_row.insert(0, "model", model_name if model_name is not None else "model")
        case_row.insert(1, "case_number", case_number)
        case_row.insert(2, "row_role", "MISCLASSIFIED_CASE")
        case_row.insert(3, "predicted_code", predicted_code)
        case_row.insert(4, "predicted_depth", predicted_depth)
        case_row.insert(5, "model_confidence", float(case.get("model_confidence", np.nan)))
        case_row.insert(6, "diagnostic_bucket", case.get("diagnostic_bucket", ""))

        if predicted_depth >= 1 and f"target_l{predicted_depth}" in records.columns:
            wrong_mask = records[f"target_l{predicted_depth}"].astype("string") == predicted_prefix
        else:
            wrong_mask = pd.Series(False, index=records.index)

        wrong_idx = records.loc[wrong_mask].index
        if idx in wrong_idx:
            wrong_idx = wrong_idx.drop(idx)
        if max_rows_per_group is not None:
            wrong_idx = wrong_idx[:max_rows_per_group]
        wrong_rows = row_context(records, wrong_idx)
        if not wrong_rows.empty:
            wrong_rows.insert(0, "row_role", "PREDICTED_WRONG_CLASS_ROW")
            wrong_rows.insert(1, "case_number", case_number)
            wrong_rows.insert(2, "case_predicted_code", predicted_code)
            wrong_rows.insert(3, "case_true_code", true_code)

        true_mask = records["target_l4"].astype("string") == true_code
        true_idx = records.loc[true_mask].index
        # Keep the case itself in the true class rows, but it is marked clearly.
        if max_rows_per_group is not None:
            true_idx = true_idx[:max_rows_per_group]
        true_rows = row_context(records, true_idx)
        if not true_rows.empty:
            true_rows.insert(0, "row_role", ["MISCLASSIFIED_CASE_IN_TRUE_CLASS" if i == idx else "TRUE_RIGHT_CLASS_ROW" for i in true_rows.index])
            true_rows.insert(1, "case_number", case_number)
            true_rows.insert(2, "case_predicted_code", predicted_code)
            true_rows.insert(3, "case_true_code", true_code)

        outputs.append(
            {
                "model": model_name if model_name is not None else "model",
                "case_number": case_number,
                "case_index": idx,
                "true_code": true_code,
                "predicted_code": predicted_code,
                "predicted_depth": predicted_depth,
                "case_row": case_row,
                "predicted_wrong_class_rows": wrong_rows,
                "true_right_class_rows": true_rows,
            }
        )

    return outputs


def make_misclassification_comparison_sample(
    predictions,
    records,
    model_name=None,
    n_cases=5,
    n_reference_rows=3,
    random_state=42,
):
    """Backward-compatible stacked table of sampled misclassification comparisons."""
    comparisons = get_misclassification_class_comparisons(
        predictions=predictions,
        records=records,
        model_name=model_name,
        n_cases=n_cases,
        max_rows_per_group=n_reference_rows,
        random_state=random_state,
    )
    frames = []
    for comp in comparisons:
        frames.extend([
            comp["case_row"],
            comp["predicted_wrong_class_rows"],
            comp["true_right_class_rows"],
        ])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=False)

def add_interpretation_columns(predictions):
    """Diagnostic columns only; not CatBoost features."""
    out = predictions.copy()

    def support_bucket(v):
        if pd.isna(v):
            return "unknown_or_unresolved_true_label"
        if v == 0:
            return "absent_in_training_fold"
        if v == 1:
            return "singleton_in_training_fold"
        if v < 4:
            return "few_samples_in_training_fold"
        return "enough_samples_in_training_fold"

    out["true_exact_support_bucket"] = out["true_support_l4"].map(support_bucket)
    out["known_x_target"] = out[TARGET_COL].astype(str).str.contains("X", case=False, na=False)

    def bucket(row):
        if row["needs_manual_tagging"]:
            return "manual_tagging"
        if row["is_safe_prediction"]:
            return "auto_tagged_safe"
        if row["known_x_target"]:
            return "unsafe_on_historical_x_target"
        if row["is_singleton_exact"]:
            return "unsafe_singleton_exact"
        if row["true_exact_support_bucket"] in [
            "absent_in_training_fold",
            "singleton_in_training_fold",
            "few_samples_in_training_fold",
        ]:
            return "unsafe_and_sparse_true_exact"
        return "unsafe_other"

    out["diagnostic_bucket"] = out.apply(bucket, axis=1)
    return out
