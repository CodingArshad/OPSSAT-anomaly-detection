# Satellite Telemetry Anomaly Detection (OPSSAT-AD)

A binary classifier that labels real satellite telemetry fragments as anomalous or nominal, built on the OPSSAT-AD dataset (actual on-orbit telemetry from ESA's OPS-SAT CubeSat). The dataset contains 2,123 labeled single-channel telemetry fragments across 9 channels, roughly 20% of them anomalous. The end goal is a complete, honest ML workflow on real flight data (load → clean → explore → split → train → evaluate), with evaluation done correctly under class imbalance — a model that always predicts "normal" scores ~80% accuracy here while detecting nothing, so accuracy alone is never the metric.

## Current Version (V4)

V4 delivers the first complete, evaluated model: a depth-constrained Decision Tree trained on the V3 feature matrix and scored the right way for an imbalanced problem (precision, recall, and F1 for the anomalous class, with a confusion matrix.)

### Features
- Verified data loading with assertions (2,123 fragments / 9 channels / ~20% anomalous) — V1
- Cleaning audit with logged decisions — V1
- Raw-fragment EDA: per-channel counts/anomaly rates, anomalous-vs-nominal fragment plots — V2
- Fixed-width feature extraction (~12–15 summary-statistic features per fragment) and feature-distribution-by-label EDA — V3
- Stratified 80/20 train/test split, fixed seed, class balance verified — V3
- `train_models()`: DecisionTreeClassifier with `max_depth` selected via 5-fold cross-validation on the training set only (over `max_depth ∈ {3,4,5,6}`), refit on the full training set with the winning depth; fitted model persisted to `results/models/` via joblib
- `evaluate_models()`: precision/recall/F1 via `classification_report` with the anomalous class as positive; confusion matrix plot; accuracy reported only as a footnote against the ~80% always-normal baseline; results sanity-checked (timeboxed to 30 min) against the reference repo's published baselines for plausibility
- Feature importance: ranked bar chart of the Decision Tree's `.feature_importances_`, with a written interpretation of which summary statistics the model actually relies on
- Experiment tracking: `results/metrics_summary.csv` (one row per run — model, hyperparameters, timestamp, precision/recall/F1) plus a JSON experiment log recording the full run configuration

## Technologies
- Python 3
- pandas, NumPy
- Matplotlib (EDA figures, confusion matrix plots, feature importance chart)
- scikit-learn (`train_test_split`, `DecisionTreeClassifier`, `StratifiedKFold`/`cross_val_score`, `classification_report`)
- joblib (model persistence)

## Project Structure
```
OPSSAT-AD/
├── data/
│   └── raw/            # dataset.csv, segments.csv (Zenodo record 12588359, not committed)
├── figures/            # EDA figures + confusion matrix + feature importance plots
├── results/
│   ├── models/          # persisted joblib models
│   ├── experiment_log/  # JSON per-run configs
│   └── metrics_summary.csv
├── src/
│   ├── load.py         # load_fragments
│   ├── clean.py        # clean_fragments
│   ├── eda.py          # run_eda (pass 1 + pass 2)
│   ├── features.py     # extract_features
│   ├── split.py        # split_data
│   ├── models.py       # train_models (Decision Tree, CV-selected max_depth, joblib persistence)
│   └── evaluate.py     # evaluate_models (metrics, feature importance, experiment log)
├── main.py             # full pipeline: load → … → train → evaluate
├── docs/
│   └── algorithm.md
└── README.md
```

## Long-Term Goals
- V5: Logistic Regression as a second (and final) model family, side-by-side comparison, and a written interpretation of which error mode dominates (missed anomalies vs. false alarms) and why that matters operationally
- V6: Final documentation pass and public GitHub release
