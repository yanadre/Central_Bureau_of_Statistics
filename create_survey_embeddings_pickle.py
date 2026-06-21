

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


STOP_WORDS = ['כנ"ל']


class SurveyTextPreprocessor(BaseEstimator, TransformerMixin):
    """Build a dense Hebrew job-description text column from allowed survey fields."""

    ALLOWED_COLUMNS = [
        "EzoAvoda",
        "SugAvoda",
        "TeurPeula",
        "TeurTafkid",
        "ShemMachlaka",
        "SugMachlaka",
        "ShemAvoda",
    ]

    def __init__(self, selected_cols=None, output_col="enriched_text"):
        if selected_cols is None:
            self.selected_cols = self.ALLOWED_COLUMNS
        else:
            invalid_cols = [c for c in selected_cols if c not in self.ALLOWED_COLUMNS]
            if invalid_cols:
                raise ValueError(
                    f"Invalid columns specified: {invalid_cols}. "
                    f"Allowed job-related text columns are: {self.ALLOWED_COLUMNS}"
                )
            self.selected_cols = selected_cols
        self.output_col = output_col

    def fit(self, X, y=None):
        return self

    def _is_missing(self, value):
        if pd.isna(value) or value is None:
            return True
        val_str = str(value).strip().lower()
        return val_str in STOP_WORDS

    def _build_dense_string(self, row):
        valid_values = []
        row_keys = row.index

        for col in self.selected_cols:
            if col in row_keys:
                val = row[col]
                if not self._is_missing(val):
                    clean_val = " ".join(str(val).split())
                    valid_values.append(clean_val)

        # Unique deduplication while preserving column order.
        # This is reproducible; list(set(...)) changes order.
        unique_values = list(dict.fromkeys(valid_values))

        joined_values = " ,".join(unique_values)
        return f"תיאור מקום עבודה ותפקיד: {joined_values}"

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        df_out = X.copy()
        df_out[self.output_col] = df_out.apply(self._build_dense_string, axis=1)
        return df_out


class SklearnSentenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="sentence-transformers/labse",
        text_col="enriched_text",
        output_emb_col="embedding",
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    ):
        self.model_name = model_name
        self.text_col = text_col
        self.output_emb_col = output_emb_col
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.model = None

    def fit(self, X, y=None):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        if self.text_col not in X.columns:
            raise KeyError(
                f"Expected text column '{self.text_col}' was not found. "
                "Ensure the preprocessor step runs before this embedder."
            )

        df_out = X.copy()
        text_inputs = df_out[self.text_col].astype("string").fillna("").values.tolist()

        embeddings = self.model.encode(
            text_inputs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        df_out[self.output_emb_col] = list(embeddings)
        return df_out


def load_survey_dataset(input_path, sheet_name="Survey Dataset"):
    input_path = Path(input_path)

    df = pd.read_excel(
        input_path,
        sheet_name=sheet_name,
        usecols="A:R",
        engine="openpyxl",
    )

    if "ID" not in df.columns:
        raise KeyError("Expected column 'ID' was not found in the survey file.")

    # Keep original ID. Remove only non-record rows with missing ID.
    df = df.loc[df["ID"].notna()].copy()
    return df


def create_embeddings_dataframe(
    input_path,
    sheet_name="Survey Dataset",
    model_name="sentence-transformers/labse",
    selected_cols=None,
    id_col="ID",
    text_col="enriched_text",
    embedding_col="embedding",
    batch_size=32,
):
    df = load_survey_dataset(input_path=input_path, sheet_name=sheet_name)

    if id_col not in df.columns:
        raise KeyError(f"Expected ID column '{id_col}' was not found.")

    preprocessor = SurveyTextPreprocessor(
        selected_cols=selected_cols,
        output_col=text_col,
    )

    embedder = SklearnSentenceTransformer(
        model_name=model_name,
        text_col=text_col,
        output_emb_col=embedding_col,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    enriched = preprocessor.fit_transform(df)
    embedded = embedder.fit(enriched).transform(enriched)

    output = embedded[[id_col, embedding_col]].copy()
    output = output.reset_index(drop=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Create survey_with_embeddings.pkl from CBS survey Excel file.")

    parser.add_argument("--input", default="DUMMY_DATASET.xlsx", help="Path to Excel file.")
    parser.add_argument("--output", default="survey_with_embeddings.pkl", help="Path to output pickle.")
    parser.add_argument("--sheet-name", default="Survey Dataset", help="Excel sheet name.")
    parser.add_argument("--model-name", default="sentence-transformers/labse", help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")

    args = parser.parse_args()

    embeddings_df = create_embeddings_dataframe(
        input_path=args.input,
        sheet_name=args.sheet_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    embeddings_df.to_pickle(output_path)

    print(f"Saved: {output_path.resolve()}")
    print(f"Rows: {len(embeddings_df)}")
    print(f"Columns: {list(embeddings_df.columns)}")
    print(embeddings_df.head())


if __name__ == "__main__":
    main()
