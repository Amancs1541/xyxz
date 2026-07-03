"""
cleaning.py

Data Cleaning Module

Responsibilities
----------------
1. Remove duplicate records
2. Handle missing values
3. Convert date column
4. Standardize categorical columns
5. Sort records
6. Generate processed dataset preview
"""

import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    logger,
    banner,
    save_figure
)


# =============================================================================
# Remove Duplicate Records
# =============================================================================

def remove_duplicates(df):

    banner("Removing Duplicates")

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    logger.info(f"Duplicates Removed : {before-after}")

    print(f"Duplicates Removed : {before-after}")

    return df


# =============================================================================
# Missing Values
# =============================================================================

def handle_missing_values(df):

    banner("Handling Missing Values")

    print(df.isnull().sum())

    # Object Columns

    object_columns = df.select_dtypes(include="object").columns

    for col in object_columns:

        df[col] = df[col].fillna("Unknown")

    # Numeric Columns

    numeric_columns = df.select_dtypes(include=["int64","float64"]).columns

    for col in numeric_columns:

        df[col] = df[col].fillna(df[col].median())

    logger.info("Missing values handled")

    return df


# =============================================================================
# Convert Date
# =============================================================================

def convert_datetime(df):

    banner("Converting Date Column")

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(subset=["date"])

    logger.info("Datetime converted")

    return df


# =============================================================================
# Standardize Categories
# =============================================================================

def standardize_categories(df):

    banner("Standardizing Categories")

    categorical = df.select_dtypes(include="object").columns

    for col in categorical:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    logger.info("Categories standardized")

    return df


# =============================================================================
# Boolean Conversion
# =============================================================================

def convert_boolean_columns(df):

    boolean_columns = [
        "to_removable_media",
        "from_removable_media"
    ]

    for col in boolean_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
                .map(
                    {
                        "TRUE":1,
                        "FALSE":0
                    }
                )
                .fillna(0)
                .astype(int)
            )

    logger.info("Boolean columns converted")

    return df


# =============================================================================
# Sort Dataset
# =============================================================================

def sort_dataset(df):

    if "date" in df.columns:

        df = df.sort_values("date")

    df = df.reset_index(drop=True)

    return df


# =============================================================================
# Dataset Preview Figure
# =============================================================================

def create_preview(df):

    banner("Creating Dataset Preview")

    preview = df.head(10)

    fig, ax = plt.subplots(figsize=(18,5))

    ax.axis("off")

    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    save_figure(
        plt,
        "processed_dataset.png"
    )

    logger.info("Dataset preview saved")


# =============================================================================
# Cleaning Pipeline
# =============================================================================

def clean_dataset(df):

    banner("DATA CLEANING")

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = convert_datetime(df)

    df = standardize_categories(df)

    df = convert_boolean_columns(df)

    df = sort_dataset(df)

    create_preview(df)

    logger.info("Cleaning Finished")

    return df


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":

    from data_loader import load_dataset

    dataset = load_dataset()

    cleaned = clean_dataset(dataset)

    print(cleaned.head())

    print(cleaned.shape)