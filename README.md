# Satellite Telemetry Anomaly Detection (OPSSAT-AD)

A binary classifier that labels real satellite telemetry fragments as anomalous or nominal. Built on the OPSSAT-AD dataset: actual on-orbit telemetry from ESA's OPS-SAT CubeSat ([Zenodo record 12588359](https://zenodo.org/records/12588359); paper [arXiv:2407.04730](https://arxiv.org/abs/2407.04730)). 2,123 labeled single-channel telemetry fragments across 9 channels, roughly 20% anomalous.

Two things this project is meant to show. First, a complete, honest ML workflow on real flight data: load, clean, explore, split, train, evaluate. Second, evaluating correctly under class imbalance. A model that always predicts "normal" gets about 80% accuracy here while catching exactly zero anomalies, so precision, recall, and F1 are the metrics from the start, not something bolted on after the fact. And to be clear about what this is: a classification problem (anomalous fragment or not), not a forecasting one.

## Current Version (V6, final)

V6 is the release version. No new functionality here, the pipeline, both models, the comparison, and the misclassification autopsy all closed out in V5. This version is just the documentation and publication pass: final README, a cleanup of the repo, and a reproducibility check run from a fresh clone.

### Features
- Verified data loading with published-fact assertions (2,123 fragments, 9 channels, ~20% anomalous) and a logged cleaning audit (V1)
- Raw-fragment EDA: per-channel fragment counts and anomaly rates, plus anomalous-vs-nominal fragment plots for all 9 channels (V2)
- Fixed-width feature extraction, about 12-15 summary-statistic features covering level, length, uniqueness, and first-difference behavior, plus encoded channel, with feature-distribution-by-label EDA to check the extraction actually separated the classes (V3)
- A stratified 80/20 train/test split with a fixed seed. Class balance verified on both sides of the split (V3)
- Decision Tree baseline. `max_depth` picked via 5-fold stratified cross-validation on the training set only (range 3-6), then evaluated on anomalous-class precision/recall/F1 with a confusion matrix. Accuracy gets reported too but only as a footnote against the ~80% always-normal baseline. Results were sanity-checked against the reference repo (kplabs-pl/OPS-SAT-AD) on a timebox, feature importance got ranked and interpreted, and the model is persisted via joblib (V4)
- Logistic Regression added (`class_weight="balanced"`, standardized features), turning the evaluation into a real two-model comparison: paired confusion matrices, paired Precision-Recall curves, and a written interpretation of which error mode dominates for each model (missed anomalies vs. false alarms) and what that costs operationally. Both models persisted via joblib (V5)
- A misclassification autopsy. Test-set errors get joined back to the raw fragment data by `fragment_id`, then plotted per channel, missed anomalies next to correctly-caught ones, with a written characterization of what kind of anomalies the summary-statistic feature set just can't see (V5)
- Every model run logged to `results/metrics_summary.csv` plus a JSON experiment log: model, hyperparameters, timestamp, metrics (V4-V5)
- The full pipeline runs end to end from `main.py` with a fixed seed. Verified reproducible from a fresh clone (V6)

The precision/recall/F1 comparison table and the autopsy write-up are the actual payoff here; both live in [`results/evaluation_report.md`](results/evaluation_report.md).

### Why this project

Built specifically as evidence for AI/ML and autonomy research outreach, not as a course assignment. A few things about it matter more than the rest:

It runs on real flight data. Actual on-orbit ESA telemetry, not a synthetic or toy benchmark, mess included. The cleaning audit is part of the deliverable, not something swept under the rug.

Evaluation under class imbalance is the actual point, not an afterthought bolted onto a model that happens to work. Anomalous-class precision/recall/F1, confusion-matrix error-mode analysis, accuracy demoted to a footnote, and an autopsy that traces false negatives back to the raw telemetry. The goal was never just measuring imbalance correctly, it was actually looking at the failure modes that imbalance produces.

And structurally, this is a fault-detection problem more than a generic classification exercise. Flagging an anomalous telemetry fragment works the same way as flagging a fault on any other sensor time series: missing a real fault is the expensive error, and summary-statistic features (level shifts, stuck values, first-difference behavior) carry most of the usable signal. What the autopsy found, specifically what kinds of faults that feature family structurally can't see, is the same kind of failure-mode reasoning that matters in fault detection generally, not just in this one dataset.

## Key Engineering Decisions
- **Split frozen before any model exists (V3).** The train/test split gets locked before `train_models` is ever called, so no model choice or hyperparameter can influence how the data was divided.
- **Imbalance-aware evaluation throughout (V4-V5).** Precision, recall, and F1 plus the confusion matrix are the reported metrics for every model. Accuracy is a footnote against the ~80% always-normal baseline. Never the headline.
- **`max_depth` picked by 5-fold stratified CV on the training set only (V4).** A principled, reproducible choice within a fixed, narrow band (3-6), not an open-ended hyperparameter search. The frozen test set never gets touched by CV.
- **`class_weight="balanced"` for Logistic Regression instead of resampling (V5).** Handles the ~80/20 imbalance directly at the model level rather than through SMOTE or manual resampling. Keeps the pipeline simple and the class distribution honest.
- **Two-model ceiling (V5).** Decision Tree and Logistic Regression, full stop, not a run through the OPSSAT-AD paper's ~30-algorithm benchmark. The skill being demonstrated is evaluating correctly under imbalance, not leaderboard placement.
- **Every run logged (V4-V5).** Model, hyperparameters, timestamp, and metrics for every fit, written to `results/metrics_summary.csv` and a JSON experiment log. Any reported number can be traced back to exactly how it was produced.

## Limitations
- **The reference-repo comparison is a sanity check, not a benchmark result.** The kplabs-pl/OPS-SAT-AD baselines come from the paper's ~30-algorithm benchmark, largely unsupervised or semi-supervised under a different evaluation protocol. This project's supervised, stratified-split numbers may not line up with those directly, and that mismatch gets noted rather than papered over (V4).
- **The feature set has a known blind spot.** The autopsy (V5) found that fixed-width summary statistics (mean, std, range, first-difference) miss anomalies that are shape- or timing-based rather than level- or spread-based. Documented, not fixed, within this project's scope.
- **Two models is not an exhaustive comparison.** Decision Tree and Logistic Regression were chosen for contrast, rule-based splits versus a linear boundary, not because they're the strongest possible classifiers for this problem. Ensembles, gradient boosting, and neural approaches are explicitly out of scope here.
- **Small dataset, single mission.** 2,123 fragments. These results describe this dataset's train/test split, not a claim about generalizing to other spacecraft or telemetry systems.

## Technologies
- Python 3
- pandas, NumPy
- Matplotlib
- scikit-learn (`train_test_split`, `DecisionTreeClassifier`, `LogisticRegression`, `StandardScaler`, `StratifiedKFold`/`cross_val_score`, `classification_report`, `precision_recall_curve`)
- joblib for model persistence

## Setup & Reproduction

1. Clone the repo and install dependencies (Python 3.9+ recommended):
   ```
   git clone https://github.com/CodingArshad/OPSSAT-anomaly-detection.git
   cd OPSSAT-anomaly-detection
   pip install -r requirements.txt
   ```
2. Download the dataset from [Zenodo record 12588359](https://zenodo.org/records/12588359) and drop `dataset.csv` and `segments.csv` into `data/raw/`. Raw data isn't committed here, see `.gitignore`:
   ```
   OPSSAT-AD/
   └── data/
       └── raw/
           ├── dataset.csv
           └── segments.csv
   ```
3. Run the pipeline from the repo root:
   ```
   python main.py
   ```
   That reproduces every figure in `figures/`, both persisted models in `results/models/`, and the metrics/experiment logs in `results/`, using a fixed random seed throughout.

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
The project is done at V6 on purpose. It closes at two models plus one misclassification autopsy, deliberately not chasing the ~30-algorithm benchmark from the OPSSAT-AD paper. A few natural extensions, explicitly out of scope here but worth noting for anyone building on this:
- Per-channel models or per-channel evaluation breakdowns (the anomaly rate isn't uniform across channels)
- Time-series-native or shape/timing-aware features, motivated directly by what the autopsy found summary statistics miss
- Picking and deploying an actual operational decision threshold for a specific cost ratio. The diagnostic PR curve and average precision added in V5 are in scope here; shipping a real operating threshold is not
