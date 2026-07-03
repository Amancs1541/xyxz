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

import pandas as pd

from logger import (
    banner,
    info
)

from utils import (
    save_csv,
    save_report
)

from visualization import (
    bar_chart,
    multi_line_plot
)

from config import (

    MODEL_SUMMARY_FILE,

    MODEL_REPORT,

    EXECUTION_REPORT,

    ANOMALY_DISTRIBUTION,

    TIME_SERIES,

    MODEL_COMPARISON

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
# Time Series Comparison
# ==========================================================

def anomaly_timeline(results):

    banner("Timeline Comparison")

    multi_line_plot(

        series=[

            results["if"]["prediction"],

            results["lof"]["prediction"],

            results["ae"]["prediction"]

        ],

        labels=[

            "Isolation Forest",

            "LOF",

            "Autoencoder"

        ],

        title="Detected Anomalies",

        xlabel="Sample",

        ylabel="Prediction",

        filename=TIME_SERIES

    )

    info("Timeline figure created")


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

        filename=MODEL_COMPARISON

    )

    info("Training comparison created")

    from datetime import datetime


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

    anomaly_timeline(

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