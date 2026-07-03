# Machine Learning-Based Anomaly Detection for Identity and Access Management Systems

## Overview

This project implements an unsupervised machine learning pipeline for detecting anomalous user behavior in Identity and Access Management (IAM) systems using the CERT Insider Threat Dataset.

Three anomaly detection algorithms are implemented and compared:

- Isolation Forest
- Local Outlier Factor (LOF)
- Deep Autoencoder

The project performs complete data preprocessing, feature engineering, model training, evaluation, and visualization.

---

# Project Structure

```
IAM-ANOMALY-DETECTION/

│
├── dataset/
│   ├── device.csv
│   ├── logon.csv
│   └── file.csv
│
├── outputs/
│   │
│   ├── figures/
│   ├── models/
│   ├── reports/
│   └── tables/
│
├── src/
│   │
│   ├── config.py
│   ├── logger.py
│   ├── utils.py
│   ├── visualization.py
│   ├── data_loader.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   │
│   └── models/
│       ├── isolation_forest_model.py
│       ├── lof_model.py
│       └── autoencoder_model.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Dataset

The implementation uses the CERT Insider Threat Dataset.

Required CSV files:

```
device.csv
logon.csv
file.csv
```

Place all files inside

```
dataset/
```

---

# Installation

Clone the repository

```bash
git clone <repository_url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Execute

```bash
python main.py
```

The complete pipeline will run automatically.

---

# Machine Learning Models

## Isolation Forest

- Unsupervised
- Tree-based anomaly detection
- Efficient for large datasets

---

## Local Outlier Factor

- Density-based anomaly detection
- Detects local deviations in data density

---

## Autoencoder

- Deep Neural Network
- Learns normal behavior
- Detects anomalies using reconstruction error

---

# Feature Engineering

The following behavioural features are generated:

- Hour
- Minute
- Day
- Month
- Weekday
- Weekend Activity
- After-hours Activity
- User Activity Count
- Device Usage Count
- File Access Count
- Unique Devices
- USB Activity
- Session Density

---

# Output Files

## Tables

```
feature_matrix.csv

if_predictions.csv

lof_predictions.csv

ae_predictions.csv

anomaly_scores.csv

model_summary.csv
```

---

## Models

```
isolation_forest.pkl

lof.pkl

autoencoder.keras

scaler.pkl
```

---

## Reports

```
execution_log.txt

execution_report.txt

model_comparison.txt
```

---

## Figures

```
dataset_preview.png

processed_dataset.png

feature_correlation.png

loss_curve.png

reconstruction_error.png

if_anomaly_scores.png

lof_anomaly_scores.png

ae_anomaly_scores.png

anomaly_distribution.png

time_series_anomalies.png

model_comparison.png
```

---

# Pipeline

```
Load Dataset
      ↓
Clean Dataset
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
Isolation Forest
      ↓
LOF
      ↓
Autoencoder
      ↓
Evaluation
      ↓
Reports & Figures
```

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Joblib

---

# Expected Outputs

The project automatically generates

- Behavioural feature matrix
- Trained machine learning models
- Anomaly predictions
- Model comparison reports
- Visualizations
- Pipeline execution logs

---

# License

This project is intended for academic and research purposes.