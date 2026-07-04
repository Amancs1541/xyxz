"""
==============================================================
preprocessing.py

Data Preprocessing Module

Responsibilities
----------------
1. Encode categorical features
2. Scale numerical features
3. Save StandardScaler
4. Generate preprocessing visualizations

==============================================================
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.config import (
    SCALER_MODEL,
    DATASET_PREVIEW,
    FEATURE_CORRELATION
)

from src.logger import (
    banner,
    info
)

from src.utils import (
    save_model
)

from src.visualization import (
    dataset_preview,
    correlation_heatmap
)


# ==========================================================
# Encode Categorical Columns
# ==========================================================

def encode_categorical(df):

    banner("Encoding Categorical Features")

    encoders = {}

    categorical_columns = [

        "user",

        "pc",

        "activity"

    ]

    for column in categorical_columns:

        if column in df.columns:

            encoder = LabelEncoder()

            df[column] = encoder.fit_transform(

                df[column].astype(str)

            )

            encoders[column] = encoder

            info(f"{column} encoded")

    return df, encoders


# ==========================================================
# Scale Features
# ==========================================================

def scale_features(df):

    banner("Scaling Features")

    scaler = StandardScaler()

    scaled = scaler.fit_transform(df)

    scaled = pd.DataFrame(

        scaled,

        columns=df.columns

    )

    return scaled, scaler


# ==========================================================
# Save Scaler
# ==========================================================

def save_scaler(scaler):

    banner("Saving Scaler")

    save_model(

        scaler,

        SCALER_MODEL

    )

    info("Scaler saved")


# ==========================================================
# Visualizations
# ==========================================================

def create_visualizations(df):

    banner("Generating Preprocessing Figures")

    dataset_preview(

        df,

        DATASET_PREVIEW

    )

    correlation_heatmap(

        df,

        FEATURE_CORRELATION

    )

    info("Preprocessing visualizations created")


# ==========================================================
# Complete Pipeline
# ==========================================================

def preprocess(df):

    banner("PREPROCESSING")

    df, encoders = encode_categorical(df)

    create_visualizations(df)

    scaled_df, scaler = scale_features(df)

    save_scaler(scaler)

    info("Preprocessing completed")

    return scaled_df, scaler, encoders


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from data_loader import load_dataset

    from cleaning import clean_dataset

    from feature_engineering import engineer_features

    dataset = load_dataset()

    dataset = clean_dataset(dataset)

    features = engineer_features(dataset)

    X, scaler, encoders = preprocess(features)

    print()

    print(X.head())

    print()

    print(X.shape)