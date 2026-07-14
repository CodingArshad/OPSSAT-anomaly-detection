# Satellite Telemetry Anomaly Detection (OPSSAT-AD)

A binary classifier that labels real satellite telemetry fragments as anomalous or nominal, built on the OPSSAT-AD dataset (actual on-orbit telemetry from ESA's OPS-SAT CubeSat). The dataset contains 2,123 labeled single-channel telemetry fragments across 9 channels, roughly 20% of them anomalous. The end goal is a complete, honest ML workflow on real flight data (load → clean → explore → split → train → evaluate), with evaluation done correctly under class imbalance (a model that always predicts "normal" scores ~80% accuracy here while detecting nothing, so accuracy alone is never the metric).

## Current Version (V5)

V5 completes the modeling work: Logistic Regression (with `class_weight="balanced"` on standardized features) joins the Decision Tree, and the evaluation becomes a genuine two-model comparison plus a written interpretation of which error mode dominates for each model (missed anomalies vs. false alarms) and why that distinction matters for a satellite. V5 also adds a misclassification autopsy: the test-set errors (false negatives especially) are joined back to the raw fragments and looked at directly, channel by channel, rather than left as numbers in a confusion matrix. Two models is the ceiling by design; the autopsy doesn't add a third model, it just makes sure the two models' failures are actually understood.

### Features
- Verified data loading with published-fact assertions (2,123 fragments / 9 channels / ~20% anomalous) — V1
- Cleaning audit with logged decisions — V1
- Raw-fragment EDA: per-channel counts/anomaly rates, anomalous-vs-nominal fragment plots — V2
- Fixed-width feature extraction and feature distribution by label EDA — V3
- Stratified 80/20 train/test split, fixed seed, class balance verified — V3
- Decision Tree baseline (constrained `max_depth` 3–6), fully evaluated — V4
- `train_models()` now also trains LogisticRegression with `class_weight="balanced"` on standardized feature; the fitted Pipeline (scaler + model) is persisted via joblib, same mechanism as the Decision Tree in V4
- `evaluate_models()` extended to a side-by-side comparison: per-model precision/recall/F1 (anomalous class positive), paired confusion matrices, accuracy footnotes vs. the ~80% baseline, sanity check against reference-repo baselines
- Written error-mode interpretation: which model misses more anomalies vs. raises more false alarms, and what each failure mode costs in an operations context (a missed anomaly is a potentially unflagged spacecraft fault; a false alarm is wasted operator attention)
- **New: paired Precision-Recall curves** for both models (`figures/pr_curves.png`) with average precision reported alongside a second, imbalance-native view of the same trade-off the confusion matrices show
- Experiment tracking extended: both models' runs appended to `results/metrics_summary.csv` and logged to `results/experiment_log/`
- **New: misclassification autopsy** : test-set predictions (both models) are joined back to `fragments_df` by `fragment_id`; false negatives and false positives are identified per channel; per-channel plots reuse the V2 `run_eda` plotting code to show missed-vs-caught anomalous fragments side by side; a written characterization documents what kind of anomalies the summary-statistic feature set fails to capture 

## Technologies
- Python 3
- pandas, NumPy
- Matplotlib (EDA figures, confusion matrix plots, missed-vs-caught fragment plots)
- scikit-learn (`train_test_split`, `DecisionTreeClassifier`, `LogisticRegression`, `StandardScaler`, `classification_report`)

## Project Structure
```
OPSSAT-AD/
├── data/
│   └── raw/            # dataset.csv, segments.csv (Zenodo record 12588359, not committed)
├── figures/            # EDA figures + paired confusion matrices + PR curves + missed-vs-caught autopsy plots
├── results/
│   ├── models/          # persisted joblib models (both)
│   ├── experiment_log/  # JSON per-run configs (both models)
│   └── metrics_summary.csv
├── src/
│   ├── load.py         # load_fragments
│   ├── clean.py        # clean_fragments
│   ├── eda.py          # run_eda (pass 1 + pass 2; reused by the autopsy plots)
│   ├── features.py     # extract_features
│   ├── split.py        # split_data
│   ├── models.py       # train_models (Decision Tree + Logistic Regression, both persisted)
│   └── evaluate.py     # evaluate_models (two-model comparison + PR curves + misclassification autopsy)
├── main.py             # full pipeline, both models, autopsy
├── docs/
│   └── algorithm.md
└── README.md
```

## Long-Term Goals
- V6: Final documentation and release pass: portfolio-facing README connecting results to their intended audience, repository cleanup, public GitHub push. No new functionality: the modeling scope, comparison, and autopsy all closed with this version.
