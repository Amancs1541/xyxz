"""
==============================================================
Local Outlier Factor (LOF)

Model Module

Outputs
-------
outputs/models/
    lof.pkl

outputs/tables/
    lof_predictions.csv

==============================================================
"""

import pandas as pd

from sklearn.neighbors import LocalOutlierFactor

from src.config import (
    LOF_NEIGHBORS,
    LOF_CONTAMINATION,
    LOF_METRIC,
    LOF_NOVELTY,
    LOF_MODEL,
    LOF_PREDICTIONS_FILE
)

from src.logger import (
    banner,
    info,
    model_started,
    model_finished,
    statistics
)

from src.utils import (
    save_csv,
    save_model
)


# ==========================================================
# Build LOF Model
# ==========================================================

def build_model():
    """
    Create Local Outlier Factor model.
    """

    model = LocalOutlierFactor(

        n_neighbors=LOF_NEIGHBORS,

        contamination=LOF_CONTAMINATION,

        metric=LOF_METRIC,

        novelty=LOF_NOVELTY

    )

    return model


# ==========================================================
# Train Model
# ==========================================================

def train(model, X):
    """
    Train LOF model.
    """

    model_started("Local Outlier Factor")

    model.fit(X)

    model_finished("Local Outlier Factor")

    return model


# ==========================================================
# Predict
# ==========================================================

def predict(model, X):
    """
    Predict anomalies.

    Returns

    0 = Normal

    1 = Anomaly
    """

    prediction = model.predict(X)

    prediction = pd.Series(prediction)

    prediction = prediction.replace({

        1: 0,

        -1: 1

    })

    return prediction


# ==========================================================
# Decision Function
# ==========================================================

def anomaly_scores(model, X):
    """
    Calculate anomaly scores.
    """

    scores = -model.decision_function(X)

    return scores


# ==========================================================
# Save Prediction CSV
# ==========================================================

def save_predictions(
    prediction,
    scores
):
    """
    Save prediction CSV.
    """

    results = pd.DataFrame({

        "prediction": prediction,

        "anomaly_score": scores

    })

    save_csv(

        results,

        LOF_PREDICTIONS_FILE

    )

    return results

    # ==========================================================
# Save Model
# ==========================================================

def save_lof_model(model):
    """
    Save trained LOF model.
    """

    save_model(

        model,

        LOF_MODEL

    )

    info("LOF model saved")


# ==========================================================
# Statistics
# ==========================================================

def print_statistics(prediction):
    """
    Display model statistics.
    """

    total = len(prediction)

    anomalies = int(prediction.sum())

    statistics(

        "Local Outlier Factor",

        total,

        anomalies

    )


# ==========================================================
# Complete Pipeline
# ==========================================================

def run_lof(X, X_predict=None):
    """
    Complete LOF Pipeline.
    """

    banner("Local Outlier Factor")

    if X_predict is None:
        X_predict = X

    model = build_model()

    model = train(

        model,

        X

    )

    prediction = predict(

        model,

        X_predict

    )

    scores = anomaly_scores(

        model,

        X_predict

    )

    save_predictions(

        prediction,

        scores

    )

    save_lof_model(

        model

    )

    print_statistics(

        prediction

    )

    info(

        "Local Outlier Factor completed"

    )

    return {

        "model": model,

        "prediction": prediction,

        "score": scores

    }


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    from data_loader import load_dataset

    from cleaning import clean_dataset

    from feature_engineering import engineer_features

    from preprocessing import preprocess

    banner("LOF TEST")

    dataset = load_dataset()

    dataset = clean_dataset(dataset)

    features = engineer_features(dataset)

    X, scaler, encoders = preprocess(

        features

    )

    results = run_lof(

        X

    )

    print()

    print("First 10 Predictions")

    print(

        results["prediction"].head(10)

    )

    print()

    print("First 10 Anomaly Scores")

    print(

        results["score"][:10]

    )

    print()

    print("LOF Completed Successfully")