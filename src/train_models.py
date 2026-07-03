"""
==============================================================
train_models.py

Train all anomaly detection models.

This module ONLY trains models.

No evaluation
No reports
No figures

==============================================================
"""

import time
import pandas as pd

from logger import (
    banner,
    info
)

from utils import (
    save_csv
)

from config import (
    ANOMALY_SCORE_FILE
)

from models.isolation_forest_model import (
    run_isolation_forest
)

from models.lof_model import (
    run_lof
)

from models.autoencoder_model import (
    run_autoencoder
)


# ==========================================================
# Isolation Forest
# ==========================================================

def train_isolation_forest(X):

    start = time.perf_counter()

    result = run_isolation_forest(X)

    result["training_time"] = (

        time.perf_counter()

        - start

    )

    return result


# ==========================================================
# Local Outlier Factor
# ==========================================================

def train_lof(X):

    start = time.perf_counter()

    result = run_lof(X)

    result["training_time"] = (

        time.perf_counter()

        - start

    )

    return result


# ==========================================================
# Autoencoder
# ==========================================================

def train_autoencoder(X):

    start = time.perf_counter()

    result = run_autoencoder(X)

    result["training_time"] = (

        time.perf_counter()

        - start

    )

    return result


# ==========================================================
# Save Scores
# ==========================================================

def save_scores(

    if_result,

    lof_result,

    ae_result

):

    banner("Saving Anomaly Scores")

    scores = pd.DataFrame({

        "Isolation Forest":

            if_result["score"],

        "Local Outlier Factor":

            lof_result["score"],

        "Autoencoder":

            ae_result["score"]

    })

    save_csv(

        scores,

        ANOMALY_SCORE_FILE

    )

    return scores


# ==========================================================
# Train All Models
# ==========================================================

def train_all_models(X):

    banner("TRAINING ALL MODELS")

    if_result = train_isolation_forest(

        X

    )

    lof_result = train_lof(

        X

    )

    ae_result = train_autoencoder(

        X

    )

    scores = save_scores(

        if_result,

        lof_result,

        ae_result

    )

    info("Model Training Finished")

    return {

        "if": if_result,

        "lof": lof_result,

        "ae": ae_result,

        "scores": scores

    }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from data_loader import load_dataset

    from cleaning import clean_dataset

    from feature_engineering import engineer_features

    from preprocessing import preprocess

    banner("TRAIN MODEL TEST")

    dataset = load_dataset()

    dataset = clean_dataset(dataset)

    features = engineer_features(

        dataset

    )

    X, scaler, encoders = preprocess(

        features

    )

    results = train_all_models(

        X

    )

    print()

    print(results.keys())