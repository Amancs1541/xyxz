"""
==============================================================
evaluate.py

Model Evaluation

Compares

• Isolation Forest
• Local Outlier Factor
• Autoencoder

using unsupervised metrics.

==============================================================
"""

from datetime import datetime

import pandas as pd

from src.logger import (
    banner,
    info
)

from src.utils import (
    load_csv,
    save_csv,
    save_report
)

from src.visualization import (
    bar_chart,
    horizontal_bar_chart,
    pca_projection
)

from src.config import (

    MODEL_SUMMARY_FILE,

    MODEL_REPORT,

    EXECUTION_REPORT,

    ANOMALY_DISTRIBUTION,

    FEATURE_MATRIX_FILE,

    MODEL_COMPARISON,

    TABLES_DIR

)


# ==========================================================
# Model Summary
# ==========================================================

def create_summary(results):

    banner("Creating Summary")

    summary = pd.DataFrame({

        "Model":[

            "Isolation Forest",

            "Local Outlier Factor",

            "Autoencoder"

        ],

        "Detected Anomalies":[

            int(

                results["if"]["prediction"].sum()

            ),

            int(

                results["lof"]["prediction"].sum()

            ),

            int(

                results["ae"]["prediction"].sum()

            )

        ],

        "Training Time (Seconds)":[

            round(

                results["if"]["training_time"],

                3

            ),

            round(

                results["lof"]["training_time"],

                3

            ),

            round(

                results["ae"]["training_time"],

                3

            )

        ]

    })

    save_csv(

        summary,

        MODEL_SUMMARY_FILE

    )

    return summary


# ==========================================================
# Feature-Based Anomaly Visualizations
# ==========================================================

def create_anomaly_feature_figures(results):

    banner("Creating Anomaly Insight Figures")

    feature_matrix_path = TABLES_DIR / FEATURE_MATRIX_FILE

    if not feature_matrix_path.exists():

        info("Feature matrix not found; skipping anomaly insight figures")

        return

    feature_matrix = load_csv(feature_matrix_path)

    if feature_matrix.empty:

        info("Feature matrix is empty; skipping anomaly insight figures")

        return

    anomaly_labels = (

        results["if"]["prediction"].astype(int)

        | results["lof"]["prediction"].astype(int)

        | results["ae"]["prediction"].astype(int)

    )

    if len(anomaly_labels) != len(feature_matrix):

        sample_size = min(len(feature_matrix), len(anomaly_labels))

        feature_matrix = feature_matrix.iloc[:sample_size].copy()

        anomaly_labels = anomaly_labels.iloc[:sample_size].copy()

    feature_matrix = feature_matrix.copy()

    feature_matrix["anomaly"] = anomaly_labels.astype(int)

    if "user" in feature_matrix.columns:

        user_counts = (

            feature_matrix.groupby("user")["anomaly"].sum()

            .sort_values(ascending=False)
            .head(20)

        )

        horizontal_bar_chart(

            labels=user_counts.index.tolist(),

            values=user_counts.values.tolist(),

            title="Top 20 Anomalous Users",

            xlabel="Anomaly Count",

            ylabel="User",

            filename="top_anomalous_users.png"

        )

    if "pc" in feature_matrix.columns:

        device_counts = (

            feature_matrix.groupby("pc")["anomaly"].sum()

            .sort_values(ascending=False)
            .head(20)

        )

        horizontal_bar_chart(

            labels=device_counts.index.tolist(),

            values=device_counts.values.tolist(),

            title="Top 20 Anomalous Devices",

            xlabel="Anomaly Count",

            ylabel="Device",

            filename="top_anomalous_devices.png"

        )

    if "activity" in feature_matrix.columns:

        activity_counts = (

            feature_matrix.groupby("activity")["anomaly"].sum()

            .sort_values(ascending=False)

        )

        bar_chart(

            labels=activity_counts.index.tolist(),

            values=activity_counts.values.tolist(),

            title="Activity-wise Anomaly Distribution",

            ylabel="Anomaly Count",

            filename="activity_distribution.png"

        )

    numeric_columns = [

        column for column in feature_matrix.columns

        if column not in {"user", "pc", "activity", "anomaly"}

        and pd.api.types.is_numeric_dtype(feature_matrix[column])

    ]

    if len(numeric_columns) >= 2:

        pca_projection(

            features=feature_matrix[numeric_columns],

            labels=feature_matrix["anomaly"],

            filename="pca_projection.png"

        )

    info("Anomaly insight figures created")


# ==========================================================
# Distribution Plot
# ==========================================================

def anomaly_distribution(results):

    banner("Anomaly Distribution")

    labels = [

        "Isolation Forest",

        "LOF",

        "Autoencoder"

    ]

    values = [

        int(results["if"]["prediction"].sum()),

        int(results["lof"]["prediction"].sum()),

        int(results["ae"]["prediction"].sum())

    ]

    bar_chart(

        labels=labels,

        values=values,

        title="Detected Anomalies",

        ylabel="Number of Anomalies",

        filename=ANOMALY_DISTRIBUTION

    )

    info("Distribution figure created")


# ==========================================================
# Training Time Plot
# ==========================================================

def training_time_plot(results):

    banner("Training Time")

    labels = [

        "Isolation Forest",

        "LOF",

        "Autoencoder"

    ]

    values = [

        results["if"]["training_time"],

        results["lof"]["training_time"],

        results["ae"]["training_time"]

    ]

    bar_chart(

        labels=labels,

        values=values,

        title="Training Time Comparison",

        ylabel="Seconds",

        filename=MODEL_COMPARISON,
        annotate=True

    )

    info("Training comparison created")


# ==========================================================
# Model Comparison Report
# ==========================================================

def create_model_report(results):

    banner("Creating Model Report")

    report = f"""
==============================================================
IAM ANOMALY DETECTION MODEL COMPARISON
==============================================================

Generated
---------
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

==============================================================
Isolation Forest
==============================================================

Detected Anomalies :
{int(results["if"]["prediction"].sum())}

Training Time :
{results["if"]["training_time"]:.3f} seconds


==============================================================
Local Outlier Factor
==============================================================

Detected Anomalies :
{int(results["lof"]["prediction"].sum())}

Training Time :
{results["lof"]["training_time"]:.3f} seconds


==============================================================
Autoencoder
==============================================================

Detected Anomalies :
{int(results["ae"]["prediction"].sum())}

Training Time :
{results["ae"]["training_time"]:.3f} seconds


==============================================================
Evaluation Method
==============================================================

This project performs UNSUPERVISED anomaly detection.

Ground truth labels are not available.

Therefore the following supervised metrics
are intentionally NOT calculated:

• Accuracy

• Precision

• Recall

• F1-score

• ROC-AUC

• Confusion Matrix


Models are compared using

• Number of detected anomalies

• Training time

• Decision scores

• Reconstruction error

==============================================================
"""

    save_report(

        report,

        MODEL_REPORT

    )

    info("Model comparison report created")


# ==========================================================
# Execution Report
# ==========================================================

def create_execution_report(results):

    banner("Creating Execution Report")

    total_samples = len(

        results["if"]["prediction"]

    )

    report = f"""
==============================================================
PIPELINE EXECUTION REPORT
==============================================================

Execution Time
--------------
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


Dataset Statistics
------------------

Samples Processed :
{total_samples}


Isolation Forest

Detected :
{int(results["if"]["prediction"].sum())}


Local Outlier Factor

Detected :
{int(results["lof"]["prediction"].sum())}


Autoencoder

Detected :
{int(results["ae"]["prediction"].sum())}


Pipeline Status
---------------

SUCCESS


Generated Files
---------------

feature_matrix.csv

if_predictions.csv

lof_predictions.csv

ae_predictions.csv

anomaly_scores.csv

model_summary.csv

model_comparison.txt

execution_report.txt

==============================================================
"""

    save_report(

        report,

        EXECUTION_REPORT

    )

    info("Execution report created")


# ==========================================================
# Console Summary
# ==========================================================

def print_summary(summary):

    banner("Evaluation Summary")

    print()

    print(summary)

    print()


# ==========================================================
# Complete Evaluation
# ==========================================================

def evaluate(results):

    banner("MODEL EVALUATION")

    summary = create_summary(

        results

    )

    anomaly_distribution(

        results

    )

    create_anomaly_feature_figures(

        results

    )

    training_time_plot(

        results

    )

    create_model_report(

        results

    )

    create_execution_report(

        results

    )

    print_summary(

        summary

    )

    info("Evaluation completed")

    return summary


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    from data_loader import load_dataset

    from cleaning import clean_dataset

    from feature_engineering import engineer_features

    from preprocessing import preprocess

    from train_models import train_all_models

    banner("EVALUATION TEST")

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

    summary = evaluate(

        results

    )

    print()

    print(summary)

    print()

    print("Evaluation Completed Successfully")