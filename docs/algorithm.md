# Satellite Telemetry Anomaly Detection V4 Algorithm

## Goal
Produce the project's first complete, evaluated classifier: a Decision Tree with constrained depth, evaluated on the frozen V3 test split using imbalance-appropriate metrics. The version's success criterion is that the evaluation is honest: anomalous-class precision/recall/F1 front and center, accuracy only as a footnote against the ~80% trivial baseline, and results sanity-checked against the reference repo's baselines to catch anything implausibly good or implausibly bad.

## Inputs
Unchanged from V1.

## Outputs
- Everything from V3
- New: trained Decision Tree model, persisted to `results/models/decision_tree.joblib` via `joblib.dump`
- New: evaluation report in `results/` : classification report (precision/recall/F1, anomalous class as positive), confusion matrix plot in `figures/`, accuracy footnote vs. the always-normal baseline, and a short (30-min-timeboxed) note on how the numbers compare to the reference repo's published baselines
- New: feature importance bar chart (`figures/feature_importance.png`) with a written interpretation
- New: `results/metrics_summary.csv` row and a JSON experiment log entry in `results/experiment_log/` (model, hyperparameters including the CV-selected `max_depth`, timestamp, metrics)

## Data Structure
Unchanged from V3.

## Core Functions

1. **`load_fragments(data_dir) -> fragments_df`** — Unchanged from V1.
2. **`clean_fragments(fragments_df) -> fragments_df`** — Unchanged from V1.
3. **`extract_features(fragments_df) -> features_df`** — Unchanged from V3.
4. **`run_eda(fragments_df, features_df) -> saved figures`** — Unchanged from V3.
5. **`split_data(features_df) -> X_train, X_test, y_train, y_test`** — Unchanged from V3.

6. **`train_models(X_train, y_train) -> {"decision_tree": model}`**
   - *Input:* training split from `split_data`.
   - *Output:* dict of trained models : in V4, the Decision Tree only. (The dict-return signature is chosen now so V5 can add a second entry without changing anything.)
   - *Process:* 5-fold cross-validation on `X_train`/`y_train` only, comparing mean F1 (anomalous class) across `max_depth ∈ {3, 4, 5, 6}`; refit a `DecisionTreeClassifier` on the full training set with the winning depth and fixed `random_state`. No standardization needed for a tree; channel encoding per V3. Persist the fitted model to `results/models/decision_tree.joblib` via `joblib.dump`.
   - *Purpose:* A constrained tree is interpretable : the top splits show which summary statistics the model actually uses to flag anomalies, which can be checked against the V2/V3 EDA intuition. CV formalizes the depth choice into a principled, reproducible pick instead of an eyeballed one, without turning into a broader hyperparameter search. The depth constraint is still the overfitting guard: 2,123 rows do not support a deep tree. Persisting the model makes every downstream number reproducible without retraining.

7. **`evaluate_models(models, X_test, y_test) -> metrics + confusion matrices + feature importance + experiment log`**
   - *Input:* trained model dict, held-out test split.
   - *Output:* per-model metrics (precision/recall/F1 for the anomalous class), confusion matrix plot, accuracy footnote, sanity-check note, feature importance chart, experiment log entry.
   - *Process:* Predict on the test set; run `classification_report` with the anomalous class as the positive class; plot the confusion matrix with labeled quadrants (missed anomalies = false negatives, false alarms = false positives); report accuracy in a footnote alongside the ~80% always-normal baseline so it isn't the headline; compare F1 against the reference repo's baselines (kplabs-pl/OPS-SAT-AD, used to check plausibility) in one sentence, timeboxed to 30 minutes (see Design Rule). Pull the Decision Tree's `.feature_importances_`, plot a ranked bar chart (`figures/feature_importance.png`), and write a short interpretation of which statistics the model actually relies on. Append a row to `results/metrics_summary.csv` and write a JSON log entry (model name, hyperparameters including the CV-selected `max_depth`, timestamp, metrics) to `results/experiment_log/`.
   - *Purpose:* This is the imbalance-evaluation skill the project exists to demonstrate. The function's structure enforces it: accuracy isn't the primary metric. Feature importance makes good on the interpretability claim `train_models` already makes but doesn't back with an artifact. The experiment log is what makes every reported number traceable back to exactly how it was produced (the reproducibility half of the project's engineering-quality bar).

## Execution Flow
1. `load_fragments` → 2. `clean_fragments` → 3. `run_eda` (pass 1) → 4. `extract_features` → 5. `run_eda` (pass 2) → 6. `split_data` → 7. `train_models` (Decision Tree) → 8. `evaluate_models` (Decision Tree).

## Design Rule
One model, fully evaluated, before any second model exists. **CV safeguards, non-negotiable:** cross-validation touches only `X_train`/`y_train`; the test set is evaluated exactly once, in `evaluate_models`, with the CV-selected depth already fixed. If the tree's recall is poor, that gets recorded and interpreted (not "fixed" by reaching for a bigger model, because the comparison model (V5) is already decided and it isn't a bigger tree). **The reference-repo sanity check is timeboxed to 30 minutes, same discipline as V5's autopsy:** the kplabs-pl/OPS-SAT-AD baselines come from the paper's ~30-algorithm benchmark, which is largely unsupervised/semi-supervised under a different evaluation protocol (the numbers may not be comparable). One sentence noting their reported range and whether the protocols even match is the deliverable. The version closes when the tree is trained, its metrics and confusion matrix are saved, and that one-sentence sanity note is written down.
