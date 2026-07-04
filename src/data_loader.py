"""
==============================================================
data_loader.py

CERT Insider Threat Dataset Loader

Loads:
    device.csv
    logon.csv
    file.csv

Returns:
    device_df
    logon_df
    file_df

==============================================================
"""

import pandas as pd

from src.config import (
    DEVICE_DATASET,
    LOGON_DATASET,
    FILE_DATASET,
    COMMON_COLUMNS
)

from src.logger import (
    banner,
    dataset_loaded,
    info
)

from src.utils import (
    load_csv
)


# ==========================================================
# Standardize Column Names
# ==========================================================

def standardize_columns(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    return df


# ==========================================================
# Convert Date Column
# ==========================================================

def convert_datetime(df):

    if "date" in df.columns:

        df["date"] = pd.to_datetime(

            df["date"],

            errors="coerce"

        )

    return df


# ==========================================================
# Validate Dataset
# ==========================================================

def validate_dataset(df, name):

    banner(f"Validating {name}")

    missing = []

    for column in COMMON_COLUMNS:

        if column not in df.columns:

            missing.append(column)

    if missing:

        raise ValueError(

            f"{name} missing columns: {missing}"

        )

    dataset_loaded(

        name,

        len(df),

        len(df.columns)

    )

    info(

        f"{name} validation completed"

    )


# ==========================================================
# Load Device Dataset
# ==========================================================

def load_device():

    banner("Loading device.csv")

    df = load_csv(

        DEVICE_DATASET

    )

    df = standardize_columns(df)

    df = convert_datetime(df)

    validate_dataset(

        df,

        "device.csv"

    )

    return df


# ==========================================================
# Load Logon Dataset
# ==========================================================

def load_logon():

    banner("Loading logon.csv")

    df = load_csv(

        LOGON_DATASET

    )

    df = standardize_columns(df)

    df = convert_datetime(df)

    validate_dataset(

        df,

        "logon.csv"

    )

    return df


# ==========================================================
# Load File Dataset
# ==========================================================

def load_file():

    banner("Loading file.csv")

    df = load_csv(

        FILE_DATASET

    )

    df = standardize_columns(df)

    df = convert_datetime(df)

    validate_dataset(

        df,

        "file.csv"

    )

    return df


# ==========================================================
# Merge Datasets
# ==========================================================

def merge_datasets(

    device_df,

    logon_df,

    file_df

):

    banner("Merging Datasets")

    merged = pd.concat(

        [

            device_df,

            logon_df,

            file_df

        ],

        ignore_index=True,

        sort=False

    )

    merged = merged.sort_values(

        "date"

    )

    merged = merged.reset_index(

        drop=True

    )

    info(

        f"Merged dataset shape : {merged.shape}"

    )

    return merged


# ==========================================================
# Complete Pipeline
# ==========================================================

def load_dataset():

    banner("Loading CERT Dataset")

    device_df = load_device()

    logon_df = load_logon()

    file_df = load_file()

    merged = merge_datasets(

        device_df,

        logon_df,

        file_df

    )

    return merged


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    dataset = load_dataset()

    print()

    print(dataset.head())

    print()

    print(dataset.shape)