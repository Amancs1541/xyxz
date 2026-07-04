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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    FIGURES_DIR,
    FIGURE_DPI,
    FIGURE_WIDTH,
    FIGURE_HEIGHT
)

from src.logger import (
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

    values_array = np.asarray(values).ravel()

    if len(values_array) > 10000:

        rng = np.random.default_rng(42)

        sampled_index = rng.choice(
            len(values_array),
            size=10000,
            replace=False
        )

        values_array = values_array[sampled_index]

    plt.hist(
        values_array,
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
    filename,
    annotate=False
):

    plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    bars = plt.bar(
        labels,
        values,
        color="#4C78A8"
    )

    plt.title(title)

    plt.ylabel(ylabel)

    plt.grid(axis="y")

    if annotate:

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    save_figure(filename)

# ==========================================================
# Horizontal Bar Chart
# ==========================================================

def horizontal_bar_chart(
    labels,
    values,
    title,
    xlabel,
    ylabel,
    filename
):

    plt.figure(figsize=(12, 8))

    order = np.argsort(values)[::-1]

    ordered_labels = np.array(labels)[order]
    ordered_values = np.array(values)[order]

    y_pos = np.arange(len(ordered_labels))

    plt.barh(
        y_pos,
        ordered_values,
        color="#4C78A8"
    )

    plt.yticks(
        y_pos,
        ordered_labels
    )

    ax = plt.gca()
    ax.invert_yaxis()

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.grid(axis="x")

    save_figure(filename)

# ==========================================================
# PCA Projection
# ==========================================================

def pca_projection(
    features,
    labels,
    filename
):

    plt.figure(figsize=(10, 8))

    sample_size = min(50000, len(features))

    if sample_size < len(features):

        sampled_index = np.random.default_rng(42).choice(
            len(features),
            size=sample_size,
            replace=False
        )

        features_sample = features.iloc[sampled_index]
        labels_sample = labels.iloc[sampled_index]

    else:

        features_sample = features
        labels_sample = labels

    numeric_features = features_sample.select_dtypes(
        include=[np.number]
    ).fillna(0)

    if numeric_features.shape[1] < 2:

        return

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(numeric_features)

    pca = PCA(n_components=2, random_state=42)

    components = pca.fit_transform(scaled_features)

    normal_mask = labels_sample.astype(int) == 0
    anomaly_mask = labels_sample.astype(int) == 1

    plt.scatter(
        components[normal_mask, 0],
        components[normal_mask, 1],
        s=10,
        alpha=0.5,
        color="tab:blue",
        label="Normal"
    )

    plt.scatter(
        components[anomaly_mask, 0],
        components[anomaly_mask, 1],
        s=10,
        alpha=0.7,
        color="tab:red",
        label="Anomaly"
    )

    plt.title("PCA Projection of Normal vs Anomalies")

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.legend()

    plt.grid(True)

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