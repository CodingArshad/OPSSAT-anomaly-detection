# Satellite Telemetry Anomaly Detection V6 Algorithm

## Goal
Ship the project. V6 adds no algorithmic capability. The goal here is simple: a stranger, specifically a professor skimming a portfolio, should be able to clone the repository, read the README, understand what was built and why the evaluation was done the way it was (including what its failure modes actually are), and reproduce every number straight from `main.py`. Restating the full project goal for the record: a binary classifier labeling real OPS-SAT telemetry fragments anomalous or nominal, demonstrating a complete, honest ML workflow on real flight data and correct evaluation under ~80/20 class imbalance, including a direct look at what gets missed and why.

## Inputs
Unchanged from V1.

## Outputs
- Everything from V5: cleaned dataset, all EDA figures including feature importance and PR curves, two trained models persisted via joblib, comparison evaluation report with error-mode interpretation, the full experiment log and metrics_summary.csv, misclassification autopsy plots and written characterization
- New: a final portfolio-facing README.md, including the section tying results back to the three target professors. Di Wu gets the real ESA flight telemetry angle, Akbas gets anomaly detection under class imbalance plus the autopsy's failure-mode reasoning (the tightest match of the three), and Phoulady gets the structural parallel to valve-sticking fault detection, sharpened by what the autopsy found about the feature set's blind spots
- New: `requirements.txt` and data-download instructions. Raw data isn't committed, just a pointer to Zenodo record 12588359
- New: a public GitHub repository, verified reproducible from a fresh clone

## Data Structure
Unchanged from V3, plus V5's in-memory error table. That table is produced transiently inside `evaluate_models`, it's not a new persisted pipeline table.

## Core Functions

1. **`load_fragments(data_dir) -> fragments_df`**, unchanged from V1.
2. **`clean_fragments(fragments_df) -> fragments_df`**, unchanged from V1.
3. **`extract_features(fragments_df) -> features_df`**, unchanged from V3.
4. **`run_eda(fragments_df, features_df) -> saved figures`**, unchanged from V3.
5. **`split_data(features_df) -> X_train, X_test, y_train, y_test`**, unchanged from V3.
6. **`train_models(X_train, y_train) -> {"decision_tree": model, "logistic_regression": model}`**, unchanged from V5.
7. **`evaluate_models(models, X_test, y_test, fragments_df) -> metrics + confusion matrices + interpretation + autopsy`**, unchanged from V5. This includes the misclassification autopsy, which got scoped into V5, not V6.

No new core functions in V6, by design. This version's work is documentation, cleanup, and release, nothing more.

## Execution Flow
1. `load_fragments` → 2. `clean_fragments` → 3. `run_eda` (pass 1) → 4. `extract_features` → 5. `run_eda` (pass 2) → 6. `split_data` → 7. `train_models` → 8. `evaluate_models` (comparison + autopsy) → 9. write the final README, verify fresh-clone reproducibility, push to public GitHub.

## Design Rule
The full stopping rule, restated one last time at close: the project stops at exactly two models (Decision Tree + Logistic Regression) plus one bounded misclassification autopsy, and deliberately does not chase the ~30-algorithm benchmark from the OPSSAT-AD paper. Evaluation stays precision/recall/F1/confusion-matrix based, never plain accuracy, because of the ~80/20 class imbalance. The autopsy's job is to explain the confusion matrix's false negatives, not spawn a whole new modeling effort. V6 follows the same discipline in miniature: documentation and release only. No "one more experiment," no last-minute model tweaks, no README claims about capabilities the code doesn't actually have. The project closes cleanly once both models are trained, evaluated, sanity-checked against the reference baselines, the error-mode interpretation is written, and the autopsy is done. Same discipline that closed AeroTrajectorySim cleanly with no scope creep.
