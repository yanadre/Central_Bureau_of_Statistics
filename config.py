# Shared configuration for the CBS occupation-coding project.
# Edit paths and modeling parameters here; the notebook and helper module import this file.

DATA_PATH = "DUMMY_DATASET.xlsx"
SHEET_NAME = "Survey Dataset"
EMBEDDINGS_PATH = "survey_with_embeddings.pkl"
EMBEDDING_COL = "embedding"
EMBEDDING_MODEL = "sentence-transformers/LaBSE"

ID_COL = "ID"
TARGET_COL = "SemelMishlachSofi"
INDUSTRY_COL = "SemelAnafSofi"

TEXT_COLS = [
    "ShemAvoda",
    "SugAvoda",
    "ShemMachlaka",
    "SugMachlaka",
    "EzoAvoda",
    "TeurPeula",
    "TeurTafkid",
]

TEXT_ORDER_FOR_BASELINE = [
    "EzoAvoda",
    "SugAvoda",
    "ShemAvoda",
    "ShemMachlaka",
    "SugMachlaka",
    "TeurPeula",
    "TeurTafkid",
]

SIMILARITY_FEATURES = [
    "sim_l1",
    "sim_l2",
    "sim_l3",
    "sim_l4",
]

# User-selected non-text fields.
# Gil was intentionally removed.
SELECTED_TABULAR_FEATURES = [
    "MenahelEtMi",
    "MakorSachar",
    "MaamadAvoda",
    "TeudaGvoha",
    "shnotlimud",
]

# Categorical variables used by CatBoost.
# TeudaGvoha is treated as categorical in this revised version.
SELECTED_CATEGORICAL_FEATURES = [
    "MakorSachar",
    "MaamadAvoda",
    "TeudaGvoha",
]

# Ordinal/numeric variables used as numeric CatBoost features.
SELECTED_NUMERIC_FEATURES = [
    "MenahelEtMi",
    "shnotlimud",
]

MODEL_FEATURES = SIMILARITY_FEATURES + SELECTED_TABULAR_FEATURES
CAT_FEATURES = SELECTED_CATEGORICAL_FEATURES

N_SPLITS = 5
RANDOM_STATE = 42

THRESHOLDS = {
    1: 0.7,
    2: 0.7,
    3: 0.7,
    4: 0.9,
}

MIN_SUPPORT_FOR_AUTO = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
}

MIN_PROBABILITY_MARGIN = 0.03
BEAM_WIDTH = 2
MAX_CHILDREN_PER_PARENT = 3

CATBOOST_PARAMS = {
    "iterations": 100,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
    "auto_class_weights": "Balanced",
}

BOW_MAX_FEATURES = 6000
TFIDF_MAX_FEATURES = 6000

MISCLASSIFIED_SAMPLE_SIZE = 5
REFERENCE_ROWS_PER_CLASS = 3
