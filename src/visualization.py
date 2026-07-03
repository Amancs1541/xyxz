"""
==============================================================
visualization.py

Central Visualization Module

All plots for the project are created here.

Used by:

cleaning.py
preprocessing.py
isolation_forest_model.py
lof_model.py
autoencoder_model.py
evaluate.py

==============================================================
"""

import matplotlib.pyplot as plt
import numpy as np

from config import (
    FIGURES_DIR,
    FIGURE_DPI,
    FIGURE_WIDTH,
    FIGURE_HEIGHT
)

from logger import (
    logger,
    file_saved
)

# ==========================================================
# Save Figure
# ==========================================================

def save_figure(filename):

    filepath = FIGURES_DIR / filename

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )

    plt.close()

    file_saved(filename)

# ==========================================================
# Dataset Preview
# ==========================================================

def dataset_preview(df, filename):

    plt.figure(figsize=(16, 5))

    plt.axis("off")

    table = plt.table(
        cellText=df.head(10).values,
        colLabels=df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)

    table.set_fontsize(8)

    table.scale(1, 1.5)

    save_figure(filename)

# ==========================================================
# Correlation Heatmap
# ==========================================================

def correlation_heatmap(df, filename):

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))

    plt.imshow(
        corr,
        interpolation="nearest"
    )

    plt.colorbar()

    plt.xticks(
        np.arange(len(corr.columns)),
        corr.columns,
        rotation=90,
        fontsize=8
    )

    plt.yticks(
        np.arange(len(corr.columns)),
        corr.columns,
        fontsize=8
    )

    plt.title("Feature Correlation")

    save_figure(filename)

# ==========================================================
# Line Plot
# ==========================================================

def line_plot(
    values,
    title,
    xlabel,
    ylabel,
    filename
):

    plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    plt.plot(values)

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.grid(True)

    save_figure(filename)

# ==========================================================
# Histogram
# ==========================================================

def histogram(
    values,
    bins,
    title,
    xlabel,
    ylabel,
    filename
):

    plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    plt.hist(
        values,
        bins=bins
    )

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.grid(True)

    save_figure(filename)

# ==========================================================
# Histogram + Threshold
# ==========================================================

def histogram_threshold(
    values,
    threshold,
    title,
    xlabel,
    ylabel,
    filename
):

    plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    plt.hist(
        values,
        bins=80
    )

    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Threshold"
    )

    plt.legend()

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.grid(True)

    save_figure(filename)

# ==========================================================
# Bar Chart
# ==========================================================

def bar_chart(
    labels,
    values,
    title,
    ylabel,
    filename
):

    plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    plt.bar(
        labels,
        values
    )

    plt.title(title)

    plt.ylabel(ylabel)

    plt.grid(axis="y")

    save_figure(filename)

# ==========================================================
# Multi Line Plot
# ==========================================================

def multi_line_plot(
    series,
    labels,
    title,
    xlabel,
    ylabel,
    filename
):

    plt.figure(
        figsize=(15, 5)
    )

    for values, label in zip(series, labels):

        plt.plot(
            values,
            label=label,
            linewidth=1
        )

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.legend()

    plt.grid(True)

    save_figure(filename)

# ==========================================================
# Training Loss
# ==========================================================

def training_loss(history, filename):

    plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    plt.plot(
        history.history["loss"],
        label="Training"
    )

    if "val_loss" in history.history:

        plt.plot(
            history.history["val_loss"],
            label="Validation"
        )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    save_figure(filename)

# ==========================================================
# Scatter Plot
# ==========================================================

def scatter_plot(
    x,
    y,
    title,
    xlabel,
    ylabel,
    filename
):

    plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    plt.scatter(
        x,
        y,
        s=5
    )

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.grid(True)

    save_figure(filename)

# ==========================================================
# Feature Importance
# ==========================================================

def feature_importance(
    feature_names,
    importance,
    filename
):

    order = np.argsort(importance)

    plt.figure(figsize=(10, 8))

    plt.barh(
        np.array(feature_names)[order],
        np.array(importance)[order]
    )

    plt.title("Feature Importance")

    save_figure(filename)