# IAM Anomaly Detection — Function Guide

## Project purpose

This project detects unusual Identity and Access Management (IAM) behaviour from the CERT `device.csv`, `logon.csv`, and `file.csv` datasets. It is an **unsupervised** machine-learning pipeline: it identifies records that look unusual without requiring labelled examples of attacks.

## Pipeline at a glance

```text
CSV activity logs
  → load and validate
  → clean and standardise
  → create behavioural features
  → encode and scale features
  → train Isolation Forest, LOF, and Autoencoder
  → compare results and save reports, tables, and figures
```

`main.py` is the entry point. Running `python main.py` calls `initialize_project()` and then `run_pipeline()`.

---

## `main.py`

| Function | What it does |
|---|---|
| `run_pipeline()` | Coordinates the complete workflow: loads data, cleans it, engineers features, preprocesses them, trains all three models, evaluates results, prints the summary, and returns it. It also records total execution time. |

---

## `src/data_loader.py` — loading and validation

| Function | What it does |
|---|---|
| `standardize_columns(df)` | Removes extra spaces and converts all column names to lowercase, so datasets use consistent names. |
| `convert_datetime(df)` | Converts the `date` column to pandas datetime values; invalid dates become missing values. |
| `validate_dataset(df, name)` | Checks that a dataset has the common required columns (`id`, `date`, `user`, `pc`, and `activity`), then logs its size. |
| `load_device()` | Loads, standardises, converts dates, and validates `device.csv`. |
| `load_logon()` | Loads, standardises, converts dates, and validates `logon.csv`. |
| `load_file()` | Loads, standardises, converts dates, and validates `file.csv`. |
| `merge_datasets(device_df, logon_df, file_df)` | Concatenates the three activity datasets, sorts the combined records by date, and resets row numbers. |
| `load_dataset()` | Runs the three individual loaders and merges their outputs into one dataset. |

## `src/cleaning.py` — data cleaning

| Function | What it does |
|---|---|
| `remove_duplicates(df)` | Removes identical duplicate records and reports how many were removed. |
| `handle_missing_values(df)` | Replaces missing text values with `Unknown` and missing numeric values with each column's median. |
| `convert_datetime(df)` | Converts `date` values to datetime format and removes rows whose dates cannot be converted. |
| `standardize_categories(df)` | Trims spaces and converts text categories to lowercase. |
| `convert_boolean_columns(df)` | Converts removable-media flags such as `TRUE`/`FALSE` into numeric `1`/`0` values. |
| `sort_dataset(df)` | Sorts records by date and resets the index. |
| `create_preview(df)` | Saves an image containing the first 10 processed rows. |
| `clean_dataset(df)` | Runs the entire cleaning sequence above and returns the cleaned dataset. |

## `src/feature_engineering.py` — behavioural features

| Function | What it does |
|---|---|
| `datetime_features(df)` | Extracts hour, minute, day, month, and weekday from the date column. |
| `weekend_feature(df)` | Creates `is_weekend`: 1 for Saturday/Sunday activity, otherwise 0. |
| `after_hours(df)` | Creates `after_hours`: 1 when activity occurs outside configured business hours, otherwise 0. |
| `user_activity(df)` | Counts the total number of records associated with each user. |
| `device_usage(df)` | Counts how many records are associated with each computer/device. |
| `file_usage(df)` | Counts access frequency for each filename; uses 0 when filename data is unavailable. |
| `unique_devices(df)` | Calculates the number of different devices used by each user. |
| `usb_activity(df)` | Combines file transfers to and from removable media into one USB-activity feature. |
| `session_density(df)` | Counts a user's events in the same day and hour, identifying unusually dense sessions. |
| `select_features(df)` | Keeps the 16 selected modelling features and saves them as `feature_matrix.csv`. |
| `engineer_features(df)` | Runs all feature-creation functions and returns the final feature matrix. |

## `src/preprocessing.py` — making features model-ready

| Function | What it does |
|---|---|
| `encode_categorical(df)` | Converts the categorical `user`, `pc`, and `activity` values into numeric labels, keeping the encoders for possible reuse. |
| `scale_features(df)` | Uses `StandardScaler` to normalise every feature to a comparable scale. |
| `save_scaler(scaler)` | Saves the fitted scaler to `outputs/models/scaler.pkl`. |
| `create_visualizations(df)` | Creates a feature-table preview and a correlation heatmap before scaling. |
| `preprocess(df)` | Runs encoding, visualisation, scaling, and scaler saving; returns scaled data, scaler, and encoders. |

## `src/train_models.py` — orchestration of model training

| Function | What it does |
|---|---|
| `prepare_training_subset(X)` | Randomly samples up to 10,000 rows for model training, reducing resource use on large datasets. |
| `train_isolation_forest(X, X_predict=None)` | Runs Isolation Forest and records its elapsed training/prediction time. |
| `train_lof(X, X_predict=None)` | Runs Local Outlier Factor and records its elapsed training/prediction time. |
| `train_autoencoder(X, X_predict=None)` | Runs the Autoencoder and records its elapsed training/prediction time. |
| `save_scores(if_result, lof_result, ae_result)` | Combines the three models' anomaly scores into `anomaly_scores.csv`. |
| `train_all_models(X)` | Trains all models on the sample, scores the full feature matrix, saves scores, and returns all results. |

---

## `src/models/isolation_forest_model.py` — Isolation Forest

Isolation Forest isolates unusual observations using random decision trees. A prediction of `1` means anomaly; `0` means normal.

| Function | What it does |
|---|---|
| `build_model()` | Creates an `IsolationForest` with the parameters defined in `config.py`. |
| `train(model, X)` | Fits the Isolation Forest to the training feature matrix. |
| `predict(model, X)` | Produces predictions and converts scikit-learn labels from `-1/1` to project labels `1/0` (anomaly/normal). |
| `anomaly_scores(model, X)` | Calculates decision scores, reversed so larger values represent greater anomaly. |
| `save_predictions(prediction, scores)` | Saves predictions and scores to `if_predictions.csv`. |
| `print_statistics(prediction)` | Logs the total samples and anomalies found. |
| `run_isolation_forest(X, X_predict=None)` | Complete Isolation Forest workflow: build, train, predict, score, save model/results, and return them. |

## `src/models/lof_model.py` — Local Outlier Factor (LOF)

LOF identifies records whose local neighbourhood density differs strongly from nearby records. A prediction of `1` means anomaly; `0` means normal.

| Function | What it does |
|---|---|
| `build_model()` | Creates the configured `LocalOutlierFactor` model. |
| `train(model, X)` | Fits LOF to the training feature matrix. |
| `predict(model, X)` | Predicts anomalies and converts `-1/1` labels to project labels `1/0`. |
| `anomaly_scores(model, X)` | Calculates reversed LOF decision scores, where larger scores are more anomalous. |
| `save_predictions(prediction, scores)` | Saves results to `lof_predictions.csv`. |
| `save_lof_model(model)` | Saves the fitted LOF model to `outputs/models/lof.pkl`. |
| `print_statistics(prediction)` | Logs total samples and detected anomalies. |
| `run_lof(X, X_predict=None)` | Complete LOF workflow: build, train, predict, score, save, and return results. |

## `src/models/autoencoder_model.py` — deep-learning Autoencoder

The Autoencoder learns to reconstruct normal feature patterns. Records with a high reconstruction error are flagged as anomalous.

| Function | What it does |
|---|---|
| `build_model(input_dimension)` | Builds and compiles a neural network with encoder layers `64→32→16→8` and mirrored decoder layers. |
| `train(model, X)` | Trains the Autoencoder to reproduce its input, uses early stopping, and saves a loss-curve figure. |
| `reconstruct(model, X)` | Produces the model's reconstructed version of the input records. |
| `reconstruction_error(X, reconstructed)` | Calculates mean squared reconstruction error for every record. |
| `calculate_threshold(errors)` | Uses the configured 95th percentile of reconstruction errors as the anomaly threshold. |
| `predict(errors, threshold)` | Marks records above the threshold as anomaly (`1`) and all others as normal (`0`). |
| `save_predictions(prediction, errors)` | Saves labels and reconstruction errors to `ae_predictions.csv`. |
| `reconstruction_plot(errors, threshold)` | Saves a histogram of reconstruction errors with the threshold marked. |
| `anomaly_score_plot(errors)` | Saves a line chart of Autoencoder reconstruction errors. |
| `save_autoencoder(model)` | Saves the trained Keras model as `autoencoder.keras`. |
| `print_statistics(prediction)` | Logs total samples and detected anomalies. |
| `run_autoencoder(X, X_predict=None)` | Complete Autoencoder workflow: build, train, reconstruct, score, label, save, visualise, and return results. |

---

## `src/evaluate.py` — comparison and reporting

| Function | What it does |
|---|---|
| `create_summary(results)` | Builds and saves `model_summary.csv`, comparing anomaly counts and elapsed times. |
| `create_anomaly_feature_figures(results)` | Combines model predictions with the feature matrix and creates top anomalous users/devices, activity distribution, and PCA figures. A record is treated as anomalous if any model flags it. |
| `anomaly_distribution(results)` | Creates a bar chart comparing the number of anomalies each model detected. |
| `training_time_plot(results)` | Creates a bar chart comparing elapsed time for each model. |
| `create_model_report(results)` | Writes the text model-comparison report. It explains why supervised metrics such as precision and recall are not calculated. |
| `create_execution_report(results)` | Writes a report confirming pipeline completion, number of samples, anomaly totals, and generated files. |
| `print_summary(summary)` | Prints the summary table to the console. |
| `evaluate(results)` | Runs every evaluation, report, and figure-generation step and returns the summary table. |

## `src/visualization.py` — shared chart functions

| Function | What it does |
|---|---|
| `save_figure(filename)` | Applies layout, saves the active matplotlib figure to `outputs/figures`, and closes it. |
| `dataset_preview(df, filename)` | Creates an image table of the first 10 rows. |
| `correlation_heatmap(df, filename)` | Creates a heatmap of numeric feature correlations. |
| `line_plot(values, title, xlabel, ylabel, filename)` | Creates a standard line graph. |
| `histogram(values, bins, title, xlabel, ylabel, filename)` | Creates a histogram. |
| `histogram_threshold(values, threshold, title, xlabel, ylabel, filename)` | Creates a histogram with a vertical anomaly threshold; samples very large inputs for readability. |
| `bar_chart(labels, values, title, ylabel, filename, annotate=False)` | Creates a vertical bar chart, optionally showing each value above its bar. |
| `horizontal_bar_chart(labels, values, title, xlabel, ylabel, filename)` | Creates a descending horizontal bar chart. |
| `pca_projection(features, labels, filename)` | Reduces numeric features to two PCA dimensions and plots normal versus anomalous records. |
| `confusion_matrix_comparison(results, filename)` | Creates one comparison matrix per model using a two-of-three-model consensus as a reference label; this is agreement analysis, not ground-truth evaluation. |
| `multi_line_plot(series, labels, title, xlabel, ylabel, filename)` | Draws multiple labelled line series on one chart. |
| `training_loss(history, filename)` | Plots Autoencoder training and validation loss by epoch. |
| `scatter_plot(x, y, title, xlabel, ylabel, filename)` | Creates a scatter plot. |
| `feature_importance(feature_names, importance, filename)` | Creates a horizontal feature-importance bar chart. |

## `src/utils.py` — reusable helpers

| Function | What it does |
|---|---|
| `create_directories()` | Creates all output folders if they do not already exist. |
| `set_random_seed(seed=RANDOM_STATE)` | Seeds Python, NumPy, and TensorFlow where available for repeatable results. |
| `load_csv(path)` | Reads a CSV file into a pandas DataFrame. |
| `save_csv(df, filename)` | Saves a DataFrame into `outputs/tables`. |
| `save_model(model, filename)` | Saves a standard Python/scikit-learn model using Joblib. |
| `save_keras_model(model, filename)` | Saves a TensorFlow/Keras model. |
| `save_report(text, filename)` | Saves text to a report file in `outputs/reports`. |
| `dataset_info(df)` | Prints shape, columns, and missing-value information. |
| `missing_value_report(df)` | Returns a table of missing-value counts and percentages. |
| `duplicate_count(df)` | Returns the number of duplicated rows. |
| `categorical_columns(df)` | Returns the names of text/object columns. |
| `numeric_columns(df)` | Returns the names of numeric columns. |
| `separator()` | Prints a console separator line. |
| `initialize_project()` | Creates output directories and sets the random seed before the pipeline starts. |

## `src/logger.py` — logging helpers

| Function | What it does |
|---|---|
| `banner(title)` | Prints and logs a visible section banner. |
| `section(title)` | Writes a formatted section heading to the log. |
| `info(message)` / `warning(message)` / `error(message)` / `exception(message)` | Write messages at the corresponding logging level. |
| `pipeline_start()` / `pipeline_end()` | Log the start and successful completion of the full pipeline. |
| `dataset_loaded(name, rows, columns)` | Logs dataset name and dimensions after loading. |
| `dataset_cleaned(rows)` | Logs the number of cleaned rows. |
| `features_created(count)` | Logs the number of selected features. |
| `model_started(model)` / `model_finished(model)` | Log model-training start and completion. |
| `file_saved(filename)` | Logs every saved output file. |
| `statistics(name, total, anomalies)` | Logs total samples and anomaly count for a model. |
| `log_exception(e)` | Writes exception information and traceback to the log. |

## `src/config.py` — settings, not functions

This file centralises project paths, file names, random seed, model hyperparameters, business-hour definitions, and figure settings. Changing a value here changes the behaviour of the relevant module without editing its function code.

## Key presentation points

- The input is raw user, device, logon, and file activity data.
- The project turns raw events into behavioural signals such as after-hours activity, USB use, activity frequency, and session density.
- It compares three complementary unsupervised approaches: tree-based isolation, local-density comparison, and neural reconstruction error.
- Results are saved as reproducible tables, trained models, charts, and text reports.
- Because the dataset pipeline does not use labelled attack outcomes, the project compares models by anomaly counts, scores, reconstruction error, runtime, and agreement—not accuracy or F1 score.
