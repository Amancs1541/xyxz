"""
==============================================================
main.py

IAM Anomaly Detection Pipeline

Master Thesis Project

Pipeline

1. Initialize Project
2. Load CERT Dataset
3. Clean Dataset
4. Feature Engineering
5. Preprocessing
6. Train Models
7. Evaluate Models
8. Save Outputs

==============================================================
"""

import time
import traceback

from src.logger import (
    banner,
    info,
    error,
    pipeline_start,
    pipeline_end
)

from src.utils import (
    initialize_project
)

from src.data_loader import (
    load_dataset
)

from src.cleaning import (
    clean_dataset
)

from src.feature_engineering import (
    engineer_features
)

from src.preprocessing import (
    preprocess
)

from src.train_models import (
    train_all_models
)

from src.evaluate import (
    evaluate
)


# ==========================================================
# Main Pipeline
# ==========================================================

def run_pipeline():

    banner("IAM ANOMALY DETECTION")

    start = time.perf_counter()

    pipeline_start()

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    info("Loading CERT datasets...")

    dataset = load_dataset()

    # ------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------

    info("Cleaning dataset...")

    dataset = clean_dataset(dataset)

    # ------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------

    info("Generating behavioural features...")

    features = engineer_features(dataset)

    # Memory cleanup
    del dataset

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------

    info("Preprocessing features...")

    X, scaler, encoders = preprocess(features)

    del features

    # ------------------------------------------------------
    # Train Models
    # ------------------------------------------------------

    info("Training machine learning models...")

    results = train_all_models(X)

    del X

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    info("Evaluating models...")

    summary = evaluate(results)

    elapsed = time.perf_counter() - start

    pipeline_end()

    banner("PIPELINE COMPLETED")

    print()

    print(summary)

    print()

    print(f"Execution Time : {elapsed:.2f} seconds")

    return summary


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    try:

        initialize_project()

        run_pipeline()

    except KeyboardInterrupt:

        print("\nExecution cancelled by user.")

    except Exception as e:

        error(str(e))

        traceback.print_exc()

        raise