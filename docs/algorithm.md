# Satellite Telemetry Anomaly Detection V6 Algorithm

## Goal
Ship the project. V6 adds no algorithmic capability — the goal is that a stranger (specifically: a professor skimming a portfolio) can clone the repository, read the README, understand what was built and why the evaluation was done the way it was (including what its failure modes are), and reproduce every number from `main.py`. The full project goal, restated for the record: a binary classifier labeling real OPS-SAT telemetry fragments anomalous vs. nominal, demonstrating a complete honest ML workflow on real flight data and correct evaluation under ~80/20 class imbalance — including direct inspection of what gets missed and why.

## Inputs
Unchanged from V1.

## Outputs
- Everything from V5 (cleaned dataset, all EDA figures including feature importance and PR curves, two trained models persisted via joblib, comparison evaluation report with error-mode interpretation, the full experiment log and metrics_summary.csv, misclassification autopsy plots and written characterization)
- New: final portfolio-facing README.md, including the section connecting results to the three target professors (Di Wu — real ESA flight telemetry; Akbas — anomaly detection under class imbalance plus the autopsy's failure-mode reasoning, tightest match; Phoulady — structural parallel to valve-sticking fault detection, sharpened by the autopsy's finding on what the feature set structurally misses)
- New: `requirements.txt` and data-download instructions (raw data not committed; pointer to Zenodo record 12588359)
- New: public GitHub repository, verified reproducible from a fresh clone

## Data Structure
Unchanged from V3 (plus V5's in-memory error table, produced transiently inside `evaluate_models` — not a new persisted pipeline table).

## Core Functions

1. **`load_fragments(data_dir) -> fragments_df`** — Unchanged from V1.
2. **`clean_fragments(fragments_df) -> fragments_df`** — Unchanged from V1.
3. **`extract_features(fragments_df) -> features_df`** — Unchanged from V3.
4. **`run_eda(fragments_df, features_df) -> saved figures`** — Unchanged from V3.
5. **`split_data(features_df) -> X_train, X_test, y_train, y_test`** — Unchanged from V3.
6. **`train_models(X_train, y_train) -> {"decision_tree": model, "logistic_regression": model}`** — Unchanged from V5.
7. **`evaluate_models(models, X_test, y_test, fragments_df) -> metrics + confusion matrices + interpretation + autopsy`** — Unchanged from V5 (this includes the misclassification autopsy, which was scoped into V5, not V6).

(No new Core Functions in V6 by design — this version's work is documentation, cleanup, and release.)

## Execution Flow
1. `load_fragments` → 2. `clean_fragments` → 3. `run_eda` (pass 1) → 4. `extract_features` → 5. `run_eda` (pass 2) → 6. `split_data` → 7. `train_models` → 8. `evaluate_models` (comparison + autopsy) → 9. Write final README, verify fresh-clone reproducibility, push to public GitHub.

## Design Rule
The full stopping rule, restated at close: the project stops at exactly two models (Decision Tree + Logistic Regression) plus one bounded misclassification autopsy, and explicitly does not chase the ~30-algorithm benchmark from the OPSSAT-AD paper. Evaluation is precision/recall/F1/confusion-matrix based, never plain accuracy, because of the ~80/20 class imbalance, and the autopsy's job is to explain the confusion matrix's false negatives, not to spawn a new modeling effort. V6 itself follows the same discipline in miniature: documentation and release only — no "one more experiment," no last-minute model tweaks, no README claims about capabilities that don't exist in the code. The project closes cleanly once both models are trained, evaluated, sanity-checked against the reference baselines, the error-mode interpretation is written, and the autopsy is complete — the same discipline that closed AeroTrajectorySim cleanly with no scope creep.
