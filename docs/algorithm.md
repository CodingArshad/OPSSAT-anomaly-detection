# Satellite Telemetry Anomaly Detection V5 Algorithm

## Goal
Complete the two-model comparison the project was scoped to: Decision Tree vs. Logistic Regression on the same frozen split, same metrics, with a written interpretation of error modes, and go one step further by looking directly at what each model actually missed. The two model families are deliberately different in kind so the comparison says something about the problem's structure, not just two scores. The misclassification autopsy closes the loop between the metrics and the raw data: it joins test-set errors back to `fragments_df` and asks, per channel, what do the missed anomalies actually look like, and why would a mean/std/range/diff-stat feature set miss them?

## Inputs
Unchanged from V1.

## Outputs
- Everything from V4
- New: trained Logistic Regression `Pipeline` (scaler + model), persisted to `results/models/logistic_regression.joblib`
- New: comparison evaluation report in `results/`, both models' precision/recall/F1 side by side, paired confusion matrix plots, and a written interpretation section covering which error mode dominates for each model and its operational meaning
- New: paired Precision-Recall curves (`figures/pr_curves.png`) with average precision for both models
- New: `results/metrics_summary.csv` and `results/experiment_log/` extended with both models' runs
- **New: misclassification autopsy report** : a table of test-set errors (fragment_id, channel, true label, each model's prediction) joined back to `fragments_df`; per-channel plots (saved to `figures/`) of false negatives alongside true positives, reusing the `run_eda` plotting code from V2; a written characterization of what kind of anomalies the summary-statistic features fail to capture, saved alongside the comparison report in `results/`

## Data Structure
Unchanged from V3, with one addition used only at evaluation time: an in-memory **error table** produced during evaluation with `fragment_id`, `channel`, `label` (true), `decision_tree_pred`, `logistic_regression_pred`, built by joining `X_test`/`y_test` predictions back to `fragments_df` on `fragment_id`. This is a working artifact of `evaluate_models`, not a new DataFrame like `fragments_df`/`features_df`.

## Core Functions

1. **`load_fragments(data_dir) -> fragments_df`** — Unchanged from V1.
2. **`clean_fragments(fragments_df) -> fragments_df`** — Unchanged from V1.
3. **`extract_features(fragments_df) -> features_df`** — Unchanged from V3.
4. **`run_eda(fragments_df, features_df) -> saved figures`** — Unchanged from V3. (Its per-channel anomalous-vs-nominal plotting logic is reused, not modified, by the new autopsy step in `evaluate_models`.)
5. **`split_data(features_df) -> X_train, X_test, y_train, y_test`** — Unchanged from V3.

6. **`train_models(X_train, y_train) -> {"decision_tree": model, "logistic_regression": model}` (extended)**
   - Decision Tree — Unchanged from V4 (including CV-selected `max_depth` and joblib persistence).
   - *New — Logistic Regression:* `LogisticRegression(class_weight="balanced")` trained on standardized features (`StandardScaler` fit on train only, applied to test at evaluation time to prevent leakage; scaler kept alongside the model in a `Pipeline`). One-hot channel encoding for the linear model. The fitted Pipeline (scaler + model together) is persisted to `results/models/logistic_regression.joblib` via `joblib.dump`.
   - *Purpose of the pairing:* two genuinely different biases, and two different imbalance strategies, so the comparison in `evaluate_models` is informative. Returns exactly two entries; no other families.

7. **`evaluate_models(models, X_test, y_test, fragments_df) -> metrics + confusion matrices + interpretation + autopsy + PR curves` (extended)**
   - *Input:* trained model dict, held-out test split, and now also `fragments_df` (needed to join errors back to the raw fragment for plotting).
   - Per-model evaluation mechanics — Unchanged from V4 (classification report with anomalous class positive, confusion matrix plot, accuracy footnote vs. ~80% baseline, timeboxed reference-repo sanity check, experiment-log entry appended to `results/metrics_summary.csv`/`results/experiment_log/`) but is now run for both models on the identical test split. Feature importance stays scoped to the Decision Tree only (V4).
   - Comparison table and written error-mode interpretation — Unchanged in structure from the original V5 scope: both models' anomalous-class precision/recall/F1 in one table; paired confusion matrices on a shared figure; a written note on which model would be preferred under which operational priority (recall-first vs. precision-first), and whether `class_weight="balanced"`'s expected recall-for-precision trade actually shows up.
   - **New — Precision-Recall curves:**
     - *Process:* For each model, compute `precision_recall_curve` and average precision on the test set (anomalous class as positive); plot both models' curves on a shared figure (`figures/pr_curves.png`) like the confusion matrices.
     - *Purpose:* Under ~80/20 imbalance, a PR curve shows the precision/recall trade-off across the full threshold range in one picture. This is diagnostic: reporting curve shape and average precision (see V6's Long-Term Goals).
   - **New — misclassification autopsy:**
     - *Process:* For each model, predict on `X_test`, identify false negatives and false positives by comparing to `y_test`. Join these (via `fragment_id`, carried through `split_data`/`features_df`) back to `fragments_df` to recover each error fragment's raw value series and channel. For each channel, plot the false negatives alongside true positives, reusing the same plotting function `run_eda` already built for anomalous-vs-nominal comparisons in V2. Write 3-4 sentences characterizing what these missed fragments have in common that the caught ones don't.
     - *Purpose:* A confusion matrix says how many anomalies were missed; the autopsy says what those misses actually look like and why the feature set missed them. This is the difference between reporting a metric and demonstrating that the metric was understood.
   - *Purpose (whole function):* This is the imbalance-evaluation skill and the failure-mode-reasoning skill the project exists to demonstrate together.

## Execution Flow
1. `load_fragments` → 2. `clean_fragments` → 3. `run_eda` (pass 1) → 4. `extract_features` → 5. `run_eda` (pass 2) → 6. `split_data` → 7. `train_models` (both models) → 8. `evaluate_models` (comparison + interpretation + misclassification autopsy).

## Design Rule
 The misclassification autopsy is an addition to evaluation depth, not a redesign: it is timeboxed (half a day), reuses `run_eda`'s existing plotting code rather than writing new visualization, and produces exactly one artifact (per-channel missed-vs-caught plots) plus one short written characterization. The version closes when both models are trained, compared on the frozen split, sanity-checked against the reference baselines, the error-mode interpretation is written, the PR-curve figure is saved, and the autopsy plots plus its 3-4 sentence characterization exist.
