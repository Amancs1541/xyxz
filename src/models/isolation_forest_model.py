"""
==============================================================
Isolation Forest Model

Outputs
-------
outputs/models/
    isolation_forest.pkl

outputs/tables/
    if_predictions.csv

==============================================================
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import (
    RANDOM_STATE,
    IF_N_ESTIMATORS,
    IF_CONTAMINATION,
    IF_MAX_SAMPLES,
    IF_BOOTSTRAP,
    IF_N_JOBS,
    ISOLATION_FOREST_MODEL,
    IF_PREDICTIONS_FILE
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
# Create Model
# ==========================================================

def build_model():

    model = IsolationForest(

        n_estimators=IF_N_ESTIMATORS,

        contamination=IF_CONTAMINATION,

        random_state=RANDOM_STATE,

        max_samples=IF_MAX_SAMPLES,

        bootstrap=IF_BOOTSTRAP,

        n_jobs=IF_N_JOBS

    )

    return model


# ==========================================================
# Train
# ==========================================================

def train(model, X):

    model_started("Isolation Forest")

    model.fit(X)

    model_finished("Isolation Forest")

    return model


# ==========================================================
# Prediction
# ==========================================================

def predict(model, X):

    prediction = model.predict(X)

    prediction = pd.Series(prediction)

    prediction = prediction.replace({

        1: 0,

        -1: 1

    })

    return prediction


# ==========================================================
# Decision Score
# ==========================================================

def anomaly_scores(model, X):

    scores = -model.decision_function(X)

    return scores


# ==========================================================
# Save Predictions
# ==========================================================

def save_predictions(prediction, scores):

    results = pd.DataFrame({

        "prediction": prediction,

        "anomaly_score": scores

    })

    save_csv(

        results,

        IF_PREDICTIONS_FILE

    )

    return results


# ==========================================================
# Statistics
# ==========================================================

def print_statistics(prediction):

    total = len(prediction)

    anomalies = int(prediction.sum())

    statistics(

        "Isolation Forest",

        total,

        anomalies

    )


# ==========================================================
# Complete Pipeline
# ==========================================================

def run_isolation_forest(X, X_predict=None):

    banner("Isolation Forest")

    if X_predict is None:
        X_predict = X

    model = build_model()

    model = train(model, X)

    prediction = predict(model, X_predict)

    scores = anomaly_scores(model, X_predict)

    save_predictions(

        prediction,

        scores

    )

    save_model(

        model,

        ISOLATION_FOREST_MODEL

    )

    print_statistics(

        prediction

    )

    info("Isolation Forest completed")

    return {

        "model": model,

        "prediction": prediction,

        "score": scores

    }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from data_loader import load_dataset
    from cleaning import clean_dataset
    from feature_engineering import engineer_features
    from preprocessing import preprocess

    dataset = load_dataset()

    dataset = clean_dataset(dataset)

    features = engineer_features(dataset)

    X, scaler, encoders = preprocess(features)

    results = run_isolation_forest(X)

    print(results["prediction"].head())