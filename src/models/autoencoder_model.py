"""
==============================================================
Autoencoder Model

Deep Learning Model for
Unsupervised IAM Anomaly Detection

Outputs
-------
outputs/models/
    autoencoder.keras

outputs/tables/
    ae_predictions.csv

==============================================================
"""

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from config import (

    AE_EPOCHS,

    AE_BATCH_SIZE,

    AE_PATIENCE,

    AE_VALIDATION_SPLIT,

    AE_OPTIMIZER,

    AE_LOSS,

    AE_THRESHOLD_PERCENTILE,

    LOSS_CURVE,

    RECONSTRUCTION_ERROR,

    AE_SCORE,

    AE_PREDICTIONS_FILE,

    AUTOENCODER_MODEL,

    MODELS_DIR

)

from logger import (

    banner,

    info,

    model_started,

    model_finished,

    statistics

)

from visualization import (

    training_loss,

    histogram_threshold,

    line_plot

)

from config import (

    LOSS_CURVE

)


# ==========================================================
# Build Autoencoder
# ==========================================================

def build_model(input_dimension):

    """
    Autoencoder Architecture

    Input

        ↓

       64

        ↓

       32

        ↓

       16

        ↓

        8

        ↓

       16

        ↓

       32

        ↓

       64

        ↓

     Output

    """

    inputs = Input(

        shape=(input_dimension,)

    )

    # ======================================================

    # Encoder

    # ======================================================

    x = Dense(

        64,

        activation="relu"

    )(inputs)

    x = Dense(

        32,

        activation="relu"

    )(x)

    x = Dense(

        16,

        activation="relu"

    )(x)

    bottleneck = Dense(

        8,

        activation="relu"

    )(x)

    # ======================================================

    # Decoder

    # ======================================================

    x = Dense(

        16,

        activation="relu"

    )(bottleneck)

    x = Dense(

        32,

        activation="relu"

    )(x)

    x = Dense(

        64,

        activation="relu"

    )(x)

    outputs = Dense(

        input_dimension,

        activation="linear"

    )(x)

    model = Model(

        inputs,

        outputs

    )

    model.compile(

        optimizer=AE_OPTIMIZER,

        loss=AE_LOSS

    )

    info("Autoencoder created")

    return model


# ==========================================================
# Train Model
# ==========================================================

def train(

    model,

    X

):

    model_started(

        "Autoencoder"

    )

    callback = EarlyStopping(

        monitor="val_loss",

        patience=AE_PATIENCE,

        restore_best_weights=True

    )

    history = model.fit(

        X,

        X,

        epochs=AE_EPOCHS,

        batch_size=AE_BATCH_SIZE,

        validation_split=AE_VALIDATION_SPLIT,

        shuffle=True,

        callbacks=[callback],

        verbose=1

    )

    training_loss(

        history,

        LOSS_CURVE

    )

    model_finished(

        "Autoencoder"

    )

    return history

    # ==========================================================
# Reconstruct Dataset
# ==========================================================

def reconstruct(
    model,
    X
):
    """
    Reconstruct input samples.
    """

    banner("Reconstructing Samples")

    reconstructed = model.predict(

        X,

        verbose=0

    )

    info("Dataset reconstructed")

    return reconstructed


# ==========================================================
# Reconstruction Error
# ==========================================================

def reconstruction_error(
    X,
    reconstructed
):
    """
    Calculate Mean Squared Error for each sample.
    """

    banner("Calculating Reconstruction Error")

    errors = np.mean(

        np.square(

            X - reconstructed

        ),

        axis=1

    )

    info("Reconstruction error calculated")

    return errors


# ==========================================================
# Threshold Selection
# ==========================================================



def calculate_threshold(errors):
    """
    Select anomaly threshold using percentile.
    """

    banner("Selecting Threshold")

    threshold = np.percentile(

        errors,

        AE_THRESHOLD_PERCENTILE

    )

    info(

        f"Threshold = {threshold:.6f}"

    )

    return threshold


# ==========================================================
# Predict Anomalies
# ==========================================================

def predict(
    errors,
    threshold
):
    """
    Prediction

    0 = Normal

    1 = Anomaly
    """

    prediction = (

        errors > threshold

    ).astype(int)

    prediction = pd.Series(

        prediction,

        name="prediction"

    )

    return prediction


# ==========================================================
# Save Prediction CSV
# ==========================================================



from utils import (

    save_csv

)

def save_predictions(
    prediction,
    errors
):

    results = pd.DataFrame({

        "prediction": prediction,

        "reconstruction_error": errors

    })

    save_csv(

        results,

        AE_PREDICTIONS_FILE

    )

    return results


# ==========================================================
# Reconstruction Error Plot
# ==========================================================




def reconstruction_plot(
    errors,
    threshold
):

    histogram_threshold(

        values=errors,

        threshold=threshold,

        title="Reconstruction Error Distribution",

        xlabel="Reconstruction Error",

        ylabel="Frequency",

        filename=RECONSTRUCTION_ERROR

    )

    info(

        "Reconstruction error figure created"

    )


# ==========================================================
# Anomaly Score Plot
# ==========================================================



def anomaly_score_plot(
    errors
):

    line_plot(

        values=errors,

        title="Autoencoder Anomaly Scores",

        xlabel="Samples",

        ylabel="Reconstruction Error",

        filename=AE_SCORE

    )

    info(

        "Autoencoder score plot created"

    )

# ==========================================================
# Save Autoencoder Model
# ==========================================================


def save_autoencoder(model):
    """
    Save trained Autoencoder model.
    """

    filepath = MODELS_DIR / AUTOENCODER_MODEL

    model.save(filepath)

    info("Autoencoder model saved")

    return filepath


# ==========================================================
# Statistics
# ==========================================================

def print_statistics(prediction):
    """
    Print anomaly statistics.
    """

    total = len(prediction)

    anomalies = int(prediction.sum())

    statistics(

        "Autoencoder",

        total,

        anomalies

    )


# ==========================================================
# Complete Pipeline
# ==========================================================

def run_autoencoder(X):
    """
    Complete Autoencoder Pipeline
    """

    banner("Autoencoder")

    input_dimension = X.shape[1]

    model = build_model(

        input_dimension

    )

    history = train(

        model,

        X

    )

    reconstructed = reconstruct(

        model,

        X

    )

    errors = reconstruction_error(

        X,

        reconstructed

    )

    threshold = calculate_threshold(

        errors

    )

    prediction = predict(

        errors,

        threshold

    )

    save_predictions(

        prediction,

        errors

    )

    reconstruction_plot(

        errors,

        threshold

    )

    anomaly_score_plot(

        errors

    )

    save_autoencoder(

        model

    )

    print_statistics(

        prediction

    )

    info("Autoencoder completed")

    return {

        "model": model,

        "prediction": prediction,

        "score": errors,

        "threshold": threshold,

        "history": history

    }


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    from data_loader import load_dataset

    from cleaning import clean_dataset

    from feature_engineering import engineer_features

    from preprocessing import preprocess

    banner("AUTOENCODER TEST")

    dataset = load_dataset()

    dataset = clean_dataset(dataset)

    features = engineer_features(dataset)

    X, scaler, encoders = preprocess(

        features

    )

    results = run_autoencoder(

        X

    )

    print()

    print("First 10 Predictions")

    print(

        results["prediction"].head(10)

    )

    print()

    print("First 10 Reconstruction Errors")

    print(

        results["score"][:10]

    )

    print()

    print(

        f"Threshold : {results['threshold']:.6f}"

    )

    print()

    print("Autoencoder Completed Successfully")