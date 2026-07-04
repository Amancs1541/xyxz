"""
==============================================================
logger.py

Central Logging Module

Every project file imports the logger from here.

Example
-------
from logger import logger
==============================================================
"""

import logging
import sys

from src.config import (
    REPORTS_DIR,
    EXECUTION_LOG,
    LINE
)

# ==========================================================
# Log File
# ==========================================================

LOG_FILE = REPORTS_DIR / EXECUTION_LOG

# ==========================================================
# Logger Instance
# ==========================================================

logger = logging.getLogger("IAM")

logger.setLevel(logging.INFO)

# Prevent duplicate handlers
logger.handlers.clear()

# ==========================================================
# Formatter
# ==========================================================

formatter = logging.Formatter(

    "%(asctime)s | %(levelname)-8s | %(message)s",

    "%Y-%m-%d %H:%M:%S"

)

# ==========================================================
# File Handler
# ==========================================================

file_handler = logging.FileHandler(

    LOG_FILE,

    mode="w",

    encoding="utf-8"

)

file_handler.setFormatter(formatter)

# ==========================================================
# Console Handler
# ==========================================================

console_handler = logging.StreamHandler(sys.stdout)

console_handler.setFormatter(formatter)

# ==========================================================
# Add Handlers
# ==========================================================

logger.addHandler(file_handler)

logger.addHandler(console_handler)

# ==========================================================
# Banner
# ==========================================================

logger.info(LINE)

logger.info("IAM Anomaly Detection Project Started")

logger.info(LINE)

# ==========================================================
# Helper Functions
# ==========================================================

def banner(title: str):

    """
    Console Banner
    """

    print()

    print(LINE)

    print(title)

    print(LINE)

    logger.info(title)


def section(title: str):

    logger.info("")

    logger.info(LINE)

    logger.info(title)

    logger.info(LINE)


def info(message: str):

    logger.info(message)


def warning(message: str):

    logger.warning(message)


def error(message: str):

    logger.error(message)


def exception(message: str):

    logger.exception(message)


# ==========================================================
# Pipeline
# ==========================================================

def pipeline_start():

    section("Pipeline Started")


def pipeline_end():

    section("Pipeline Finished Successfully")


# ==========================================================
# Dataset Logging
# ==========================================================

def dataset_loaded(name, rows, columns):

    logger.info(

        f"{name} Loaded "

        f"({rows} rows × {columns} columns)"

    )


def dataset_cleaned(rows):

    logger.info(

        f"Dataset Cleaned "

        f"({rows} rows)"

    )


# ==========================================================
# Feature Logging
# ==========================================================

def features_created(count):

    logger.info(

        f"{count} Features Created"

    )


# ==========================================================
# Model Logging
# ==========================================================

def model_started(model):

    section(f"Training {model}")


def model_finished(model):

    logger.info(

        f"{model} Training Completed"

    )


# ==========================================================
# Save Logging
# ==========================================================

def file_saved(filename):

    logger.info(

        f"Saved : {filename}"

    )


# ==========================================================
# Statistics
# ==========================================================

def statistics(name, total, anomalies):

    logger.info(

        f"{name}"

        f" | Samples={total}"

        f" | Anomalies={anomalies}"

    )


# ==========================================================
# Errors
# ==========================================================

def log_exception(e):

    logger.exception(str(e))