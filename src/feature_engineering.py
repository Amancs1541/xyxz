"""
==============================================================
feature_engineering.py

IAM Behavioural Feature Engineering

Creates behavioural features from:

device.csv
logon.csv
file.csv

==============================================================
"""

import pandas as pd

from src.logger import (
    banner,
    info,
    features_created
)

from src.config import (
    BUSINESS_START,
    BUSINESS_END,
    WEEKEND,
    FEATURE_MATRIX_FILE
)

from src.utils import (
    save_csv
)


# ==========================================================
# Date Features
# ==========================================================

def datetime_features(df):

    banner("Datetime Features")

    df["hour"] = df["date"].dt.hour

    df["minute"] = df["date"].dt.minute

    df["day"] = df["date"].dt.day

    df["month"] = df["date"].dt.month

    df["weekday"] = df["date"].dt.weekday

    return df


# ==========================================================
# Weekend Activity
# ==========================================================

def weekend_feature(df):

    banner("Weekend Feature")

    df["is_weekend"] = (

        df["weekday"]

        .isin(WEEKEND)

        .astype(int)

    )

    return df


# ==========================================================
# After Hours Login
# ==========================================================

def after_hours(df):

    banner("After Hours")

    df["after_hours"] = (

        (

            (df["hour"] < BUSINESS_START)

            |

            (df["hour"] > BUSINESS_END)

        )

        .astype(int)

    )

    return df


# ==========================================================
# User Activity Count
# ==========================================================

def user_activity(df):

    banner("User Activity")

    df["user_activity_count"] = (

        df

        .groupby("user")["id"]

        .transform("count")

    )

    return df


# ==========================================================
# Device Usage Count
# ==========================================================

def device_usage(df):

    banner("Device Usage")

    df["device_usage_count"] = (

        df

        .groupby("pc")["id"]

        .transform("count")

    )

    return df


# ==========================================================
# File Access Count
# ==========================================================

def file_usage(df):

    banner("File Usage")

    if "filename" in df.columns:

        df["file_access_count"] = (

            df

            .groupby("filename")["id"]

            .transform("count")

        )

    else:

        df["file_access_count"] = 0

    return df


# ==========================================================
# Unique Devices Per User
# ==========================================================

def unique_devices(df):

    banner("Unique Devices")

    df["unique_devices"] = (

        df

        .groupby("user")["pc"]

        .transform("nunique")

    )

    return df


# ==========================================================
# USB Activity
# ==========================================================

def usb_activity(df):

    banner("USB Activity")

    if "to_removable_media" not in df.columns:

        df["to_removable_media"] = 0

    if "from_removable_media" not in df.columns:

        df["from_removable_media"] = 0

    df["usb_activity"] = (

        df["to_removable_media"]

        +

        df["from_removable_media"]

    )

    return df


# ==========================================================
# Session Density
# ==========================================================

def session_density(df):

    banner("Session Density")

    session = (

        df

        .groupby(

            [

                "user",

                "day",

                "hour"

            ]

        )["id"]

        .transform("count")

    )

    df["session_density"] = session

    return df


# ==========================================================
# Select Features
# ==========================================================

def select_features(df):

    banner("Selecting Features")

    columns = [

        "user",

        "pc",

        "activity",

        "hour",

        "minute",

        "day",

        "month",

        "weekday",

        "is_weekend",

        "after_hours",

        "user_activity_count",

        "device_usage_count",

        "file_access_count",

        "unique_devices",

        "usb_activity",

        "session_density"

    ]

    features = df[columns].copy()

    save_csv(

        features,

        FEATURE_MATRIX_FILE

    )

    features_created(

        len(features.columns)

    )

    return features


# ==========================================================
# Pipeline
# ==========================================================

def engineer_features(df):

    banner("FEATURE ENGINEERING")

    df = datetime_features(df)

    df = weekend_feature(df)

    df = after_hours(df)

    df = user_activity(df)

    df = device_usage(df)

    df = file_usage(df)

    df = unique_devices(df)

    df = usb_activity(df)

    df = session_density(df)

    feature_matrix = select_features(df)

    info("Feature engineering completed")

    return feature_matrix


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from data_loader import load_dataset

    from cleaning import clean_dataset

    data = load_dataset()

    data = clean_dataset(data)

    features = engineer_features(data)

    print(features.head())

    print(features.shape)