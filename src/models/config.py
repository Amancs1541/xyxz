"""
==========================================================
Project Configuration
==========================================================

Master Thesis

Machine Learning-Based Anomaly Detection for
Identity and Access Management Systems

Author : Your Name
Version : 1.0
==========================================================
"""

from pathlib import Path

# ==========================================================
# PROJECT INFO
# ==========================================================

PROJECT_NAME = "IAM Anomaly Detection"

VERSION = "1.0"

AUTHOR = "Master Thesis"

# ==========================================================
# ROOT PATHS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT_DIR / "src"

DATASET_DIR = ROOT_DIR / "dataset"

OUTPUT_DIR = ROOT_DIR / "outputs"

FIGURES_DIR = OUTPUT_DIR / "figures"

MODELS_DIR = OUTPUT_DIR / "models"

TABLES_DIR = OUTPUT_DIR / "tables"

REPORTS_DIR = OUTPUT_DIR / "reports"

# ==========================================================
# DATASET FILES
# ==========================================================

DEVICE_DATASET = DATASET_DIR / "device.csv"

LOGON_DATASET = DATASET_DIR / "logon.csv"

FILE_DATASET = DATASET_DIR / "file.csv"

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

for directory in [

    OUTPUT_DIR,

    FIGURES_DIR,

    MODELS_DIR,

    TABLES_DIR,

    REPORTS_DIR

]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# ISOLATION FOREST
# ==========================================================

IF_N_ESTIMATORS = 200

IF_CONTAMINATION = 0.02

IF_MAX_SAMPLES = "auto"

IF_BOOTSTRAP = False

IF_N_JOBS = -1

# ==========================================================
# LOCAL OUTLIER FACTOR
# ==========================================================

LOF_NEIGHBORS = 20

LOF_CONTAMINATION = 0.02

LOF_METRIC = "euclidean"

LOF_NOVELTY = True

# ==========================================================
# AUTOENCODER
# ==========================================================

AE_EPOCHS = 50

AE_BATCH_SIZE = 256

AE_VALIDATION_SPLIT = 0.20

AE_PATIENCE = 5

AE_THRESHOLD_PERCENTILE = 95

AE_OPTIMIZER = "adam"

AE_LOSS = "mse"

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

BUSINESS_START = 7

BUSINESS_END = 19

WEEKEND = [5, 6]

# ==========================================================
# FIGURES
# ==========================================================

FIGURE_DPI = 300

FIGURE_WIDTH = 12

FIGURE_HEIGHT = 6

PREVIEW_ROWS = 10

# ==========================================================
# MODELS
# ==========================================================

ISOLATION_FOREST_MODEL = "isolation_forest.pkl"

LOF_MODEL = "lof.pkl"

AUTOENCODER_MODEL = "autoencoder.keras"

SCALER_MODEL = "scaler.pkl"

# ==========================================================
# TABLE FILES
# ==========================================================

FEATURE_MATRIX_FILE = "feature_matrix.csv"

IF_PREDICTIONS_FILE = "if_predictions.csv"

LOF_PREDICTIONS_FILE = "lof_predictions.csv"

AE_PREDICTIONS_FILE = "ae_predictions.csv"

ANOMALY_SCORE_FILE = "anomaly_scores.csv"

MODEL_SUMMARY_FILE = "model_summary.csv"

# ==========================================================
# REPORT FILES
# ==========================================================

EXECUTION_LOG = "execution_log.txt"

EXECUTION_REPORT = "execution_report.txt"

MODEL_REPORT = "model_comparison.txt"

# ==========================================================
# FIGURE FILES
# ==========================================================

DATASET_PREVIEW = "dataset_preview.png"

PROCESSED_DATASET = "processed_dataset.png"

FEATURE_CORRELATION = "feature_correlation.png"

IF_SCORE = "if_anomaly_scores.png"

LOF_SCORE = "lof_anomaly_scores.png"

AE_SCORE = "ae_anomaly_scores.png"

LOSS_CURVE = "loss_curve.png"

RECONSTRUCTION_ERROR = "reconstruction_error.png"

ANOMALY_DISTRIBUTION = "anomaly_distribution.png"

TIME_SERIES = "time_series_anomalies.png"

MODEL_COMPARISON = "model_comparison.png"

# ==========================================================
# MODEL NAMES
# ==========================================================

MODEL_NAMES = [

    "Isolation Forest",

    "Local Outlier Factor",

    "Autoencoder"

]

# ==========================================================
# SUPPORTED DATASET COLUMNS
# ==========================================================

COMMON_COLUMNS = [

    "id",

    "date",

    "user",

    "pc",

    "activity"

]

FILE_COLUMNS = [

    "filename",

    "to_removable_media",

    "from_removable_media",

    "content"

]

DEVICE_COLUMNS = [

    "file_tree"

]

# ==========================================================
# CONSOLE
# ==========================================================

LINE = "=" * 80