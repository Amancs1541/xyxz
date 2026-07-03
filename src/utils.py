"""
==============================================================
utils.py

Project Utility Functions

This module provides reusable helper functions for:

- Random seed initialization
- Saving CSV files
- Saving models
- Saving text reports
- Loading CSV files
- Directory management
- Basic dataset information

==============================================================
"""

import random
import joblib
import numpy as np
import pandas as pd

from config import (
    RANDOM_STATE,
    OUTPUT_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    TABLES_DIR,
    REPORTS_DIR
)

from logger import logger, file_saved

# ==========================================================
# Create Output Directories
# ==========================================================

def create_directories():
    """
    Create project output directories.
    Safe to call multiple times.
    """

    directories = [

        OUTPUT_DIR,

        FIGURES_DIR,

        MODELS_DIR,

        TABLES_DIR,

        REPORTS_DIR

    ]

    for directory in directories:

        directory.mkdir(

            parents=True,

            exist_ok=True

        )

# ==========================================================
# Random Seed
# ==========================================================

def set_random_seed(seed=RANDOM_STATE):
    """
    Make experiments reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    try:

        import tensorflow as tf

        tf.random.set_seed(seed)

    except Exception:

        pass

    logger.info(f"Random Seed = {seed}")

# ==========================================================
# Load CSV
# ==========================================================

def load_csv(path):
    """
    Load CSV dataset.
    """

    df = pd.read_csv(

        path,

        low_memory=False

    )

    return df

# ==========================================================
# Save CSV
# ==========================================================

def save_csv(df, filename):
    """
    Save DataFrame to outputs/tables
    """

    filepath = TABLES_DIR / filename

    df.to_csv(

        filepath,

        index=False

    )

    file_saved(filename)

    return filepath

# ==========================================================
# Save Model
# ==========================================================

def save_model(model, filename):
    """
    Save ML model using joblib.
    """

    filepath = MODELS_DIR / filename

    joblib.dump(

        model,

        filepath

    )

    file_saved(filename)

    return filepath

# ==========================================================
# Save Keras Model
# ==========================================================

def save_keras_model(model, filename):
    """
    Save TensorFlow model.
    """

    filepath = MODELS_DIR / filename

    model.save(filepath)

    file_saved(filename)

    return filepath

# ==========================================================
# Save Text Report
# ==========================================================

def save_report(text, filename):
    """
    Save text report.
    """

    filepath = REPORTS_DIR / filename

    with open(

        filepath,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(text)

    file_saved(filename)

    return filepath

# ==========================================================
# Dataset Information
# ==========================================================

def dataset_info(df):
    """
    Print dataset summary.
    """

    print("\nShape")

    print(df.shape)

    print("\nColumns")

    print(df.columns.tolist())

    print("\nMissing Values")

    print(df.isnull().sum())

# ==========================================================
# Missing Values
# ==========================================================

def missing_value_report(df):
    """
    Return missing value report.
    """

    report = pd.DataFrame({

        "Missing Values":

            df.isnull().sum(),

        "Percentage":

            (
                df.isnull().sum()

                / len(df)

            ) * 100

    })

    return report

# ==========================================================
# Duplicate Report
# ==========================================================

def duplicate_count(df):
    """
    Return duplicate count.
    """

    return df.duplicated().sum()

# ==========================================================
# Object Columns
# ==========================================================

def categorical_columns(df):

    return list(

        df.select_dtypes(

            include="object"

        ).columns

    )

# ==========================================================
# Numeric Columns
# ==========================================================

def numeric_columns(df):

    return list(

        df.select_dtypes(

            include=np.number

        ).columns

    )

# ==========================================================
# Print Separator
# ==========================================================

def separator():

    print("-" * 80)

# ==========================================================
# Initialize Project
# ==========================================================

def initialize_project():
    """
    Initialize project.
    """

    create_directories()

    set_random_seed()

    logger.info("Project Initialized")