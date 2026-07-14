# Satellite Telemetry Anomaly Detection (OPSSAT-AD)

A binary classifier that labels real satellite telemetry fragments as anomalous or nominal, built on the OPSSAT-AD dataset — actual on-orbit telemetry from ESA's OPS-SAT CubeSat ([Zenodo record 12588359](https://zenodo.org/records/12588359); paper [arXiv:2407.04730](https://arxiv.org/abs/2407.04730)). The dataset contains 2,123 labeled single-channel telemetry fragments across 9 channels, roughly 20% anomalous. The project demonstrates two specific engineering skills: running a complete, honest ML workflow on real flight data (load → clean → explore → split → train → evaluate), and evaluating correctly under class imbalance — a model that always predicts "normal" scores ~80% accuracy here while detecting nothing, so this evaluation is built on precision, recall, and F1 from the start. This is a classification problem (anomalous fragment vs. not), not forecasting.

## Current Version (V6 — final)

V6 is the release version. No new functionality — the pipeline, both models, the comparison, and the misclassification autopsy all closed in V5. This version is the documentation and publication pass: final README, repository cleanup, and a reproducibility check from a fresh clone.

### Features
- Verified data loading with published-fact assertions (2,123 fragments / 9 channels / ~20% anomalous) and a logged cleaning audit — V1
- Raw-fragment EDA: per-channel fragment counts and anomaly rates; anomalous-vs-nominal fragment plots for all 9 channels — V2
- Fixed-width feature extraction (~12–15 summary-statistic features: level stats, length, uniqueness ratios, first-difference stats, encoded channel) with feature-distribution-by-label EDA — V3
- Stratified 80/20 train/test split, fixed seed, class balance verified on both splits — V3
- Decision Tree baseline, `max_depth` selected via 5-fold stratified cross-validation on the training set (3–6 range), evaluated on anomalous-class precision/recall/F1 with confusion matrix; accuracy reported only as a footnote against the ~80% always-normal baseline; results sanity-checked (timeboxed) against the reference repo (kplabs-pl/OPS-SAT-AD); feature importance ranked and interpreted; model persisted via joblib — V4
- Logistic Regression (`class_weight="balanced"`, standardized features) and a two-model comparison with paired confusion matrices, paired Precision-Recall curves, and a written interpretation of dominant error modes (missed anomalies vs. false alarms) and their operational cost; both models persisted via joblib — V5
- Misclassification autopsy: test-set errors joined back to the raw fragment data by `fragment_id`, per-channel plots of missed anomalies alongside correctly-caught ones, and a written characterization of what kind of anomalies the summary-statistic feature set fails to capture — V5
- Every model run logged to `results/metrics_summary.csv` and a JSON experiment log — model, hyperparameters, timestamp, metrics — V4–V5
- Full pipeline reproducible end-to-end from `main.py` with a fixed seed — V6 (verified from a fresh clone)

Full results, including the precision/recall/F1 comparison table and the misclassification autopsy's write-up, are in [`results/evaluation_report.md`](results/evaluation_report.md).

### Why this project

This project was built as evidence for research outreach, and its design maps onto three research programs:

- **Di Wu** — the data is real ESA flight telemetry, not a synthetic or toy benchmark: every result here was produced against actual on-orbit spacecraft behavior, including its mess (the cleaning audit is part of the deliverable, not a footnote).
- **Akbas** — the tightest match: the core of the project is anomaly detection under class imbalance, and the evaluation methodology (anomalous-class precision/recall/F1, confusion-matrix error-mode analysis, accuracy demoted to a baseline footnote, and the misclassification autopsy that traces false negatives back to the raw telemetry) is the project's central demonstrated skill — it shows not just that imbalance was measured correctly, but that the resulting failure modes were actually investigated.
- **Phoulady** — structurally, flagging anomalous telemetry fragments parallels valve-sticking fault detection: a labeled fault-vs-nominal classification on sensor time series, where the operationally expensive error is the missed fault, and where summary-statistic features (level shifts, stuck/duplicate values, first-difference behavior) carry the signal. The autopsy's finding — what kind of faults a summary-statistic feature set structurally cannot see — is exactly the kind of failure-mode reasoning that matters in a fault-detection context, not just a modeling exercise.

## Key Engineering Decisions
- **Split frozen before any model exists (V3)** — the train/test split is locked before `train_models` is ever called, so no model or hyperparameter choice can influence how the data was divided.
- **Imbalance-aware evaluation throughout (V4–V5)** — precision/recall/F1 and the confusion matrix are the reported metrics for every model; plain accuracy is demoted to a footnote against the ~80% always-normal baseline, never the headline.
- **`max_depth` selected by 5-fold stratified cross-validation on the training set only (V4)** — a principled, reproducible pick within a fixed, narrow band (3–6), not an open hyperparameter search; the frozen test set is never touched by CV.
- **`class_weight="balanced"` for Logistic Regression, not resampling (V5)** — handles the ~80/20 imbalance explicitly at the model level rather than via SMOTE or manual resampling, keeping the pipeline simple and the class distribution honest.
- **Two-model ceiling (V5)** — Decision Tree and Logistic Regression only, not a run through the OPSSAT-AD paper's ~30-algorithm benchmark; the skill being demonstrated is evaluating correctly under imbalance, not leaderboard placement.
- **Every run logged (V4–V5)** — model, hyperparameters, timestamp, and metrics for every model fit are written to `results/metrics_summary.csv` and a JSON experiment log, so any reported number can be traced back to exactly how it was produced.

## Limitations
- **Reference-repo comparison is a sanity check, not a benchmark result** — the kplabs-pl/OPS-SAT-AD baselines come from the paper's ~30-algorithm benchmark, largely unsupervised/semi-supervised under a different evaluation protocol; this project's supervised, stratified-split numbers may not be directly comparable, and that mismatch is noted rather than resolved (V4).
- **The feature set has a known blind spot** — the misclassification autopsy (V5) found that fixed-width summary statistics (mean/std/range/first-difference) miss anomalies that are shape- or timing-based rather than level- or spread-based; this is documented, not fixed, within this project's scope.
- **Two models, not an exhaustive comparison** — Decision Tree and Logistic Regression were chosen deliberately for contrast (rule-based splits vs. a linear boundary), not because they're the strongest possible classifiers for this problem; ensembles, gradient boosting, and neural approaches are explicitly out of scope here.
- **Small dataset (2,123 fragments) from a single mission** — results describe this dataset's train/test split, not a claim about generalization to other spacecraft or telemetry systems.

## Technologies
- Python 3
- pandas, NumPy
- Matplotlib
- scikit-learn (`train_test_split`, `DecisionTreeClassifier`, `LogisticRegression`, `StandardScaler`, `StratifiedKFold`/`cross_val_score`, `classification_report`, `precision_recall_curve`)
- joblib (model persistence)

## Setup & Reproduction

1. Clone the repository and install dependencies (Python 3.9+ recommended):
   ```
   git clone https://github.com/CodingArshad/OPSSAT-anomaly-detection.git
   cd OPSSAT-anomaly-detection
   pip install -r requirements.txt
   ```
2. Download the dataset from [Zenodo record 12588359](https://zenodo.org/records/12588359) and place `dataset.csv` and `segments.csv` in `data/raw/` (raw data isn't committed to this repo — see `.gitignore`):
   ```
   OPSSAT-AD/
   └── data/
       └── raw/
           ├── dataset.csv
           └── segments.csv
   ```
3. Run the full pipeline from the repository root:
   ```
   python main.py
   ```
   This reproduces every figure in `figures/`, both persisted models in `results/models/`, and the metrics/experiment logs in `results/`, using a fixed random seed throughout.

## Project Structure
```
OPSSAT-AD/
├── data/
│   └── raw/            # dataset.csv, segments.csv (Zenodo record 12588359, not committed)
├── figures/            # EDA figures + feature importance + confusion matrices + PR curves + misclassification autopsy plots
├── results/
│   ├── models/          # persisted joblib models (Decision Tree + Logistic Regression Pipeline)
│   ├── experiment_log/  # JSON per-run configs
│   ├── metrics_summary.csv
│   └── evaluation_report.md
├── src/
│   ├── load.py         # load_fragments
│   ├── clean.py        # clean_fragments
│   ├── eda.py          # run_eda (pass 1 + pass 2; reused by the autopsy plots)
│   ├── features.py     # extract_features
│   ├── split.py        # split_data
│   ├── models.py       # train_models (both models, CV-selected max_depth, joblib persistence)
│   └── evaluate.py     # evaluate_models (comparison + feature importance + PR curves + misclassification autopsy + experiment logging)
├── main.py             # full reproducible pipeline
├── requirements.txt
├── docs/
│   └── algorithm.md
└── README.md
```

## Long-Term Goals
The project is complete at V6 by design — it closes at two models plus the misclassification autopsy, deliberately not chasing the ~30-algorithm benchmark from the OPSSAT-AD paper, the same clean-close discipline that finished AeroTrajectorySim at V6. Natural extensions, explicitly out of scope but noted for anyone building on this:
- Per-channel models or per-channel evaluation breakdowns (the anomaly rate varies by channel)
- Time-series-native or shape/timing-aware features, motivated directly by the autopsy's finding about what summary statistics miss
- Selecting and deploying an operational decision threshold for a specific cost ratio (the diagnostic Precision-Recall curve and average precision added in V5 are in scope; choosing and shipping an actual operating threshold is not)
