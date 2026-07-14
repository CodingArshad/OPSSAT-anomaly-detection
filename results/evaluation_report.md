# V5 Evaluation Report: Decision Tree vs. Logistic Regression

Both models evaluated once, on the identical frozen V3 test split (425 fragments, 87 anomalous, 20.47% positive rate). Anomalous class treated as positive throughout.

## Comparison table (anomalous class)

| Model                | Precision | Recall | F1   | Accuracy | Baseline Accuracy |
|-----------------------|-----------|--------|------|----------|--------------------|
| Decision Tree          | 0.77      | 0.66   | 0.71 | 88.94%   | 79.53%             |
| Logistic Regression    | 0.55      | 0.74   | 0.63 | 82.12%   | 79.53%             |

Both models beat the always-predict-nominal baseline on accuracy, but per the project's own Design Rule, accuracy is a footnote here. The precision/recall split is the actual finding.

## Which error mode dominates for each model

**Decision Tree**: higher precision (0.77) than recall (0.66), it leans toward missed anomalies (false negatives) over false alarms. When it flags a fragment as anomalous, it's usually right. Its weakness is the anomalies it stays quiet about, roughly 1 in 3 real anomalies (30 of 87) go uncaught.

**Logistic Regression**: the opposite profile, recall (0.74) clearly exceeds precision (0.55). It catches more real anomalies than the tree, but at the cost of raising more false alarms. Close to half of its "anomalous" flags turn out to be nominal fragments.

## Does `class_weight="balanced"`'s expected trade show up?

Yes, clearly. `class_weight="balanced"` reweights training so misclassifying the minority (anomalous) class costs more, and the expectation was exactly what shows up: recall rises relative to a model without explicit imbalance handling, at precision's expense. The Decision Tree, which has no equivalent explicit reweighting mechanism (its imbalance handling is indirect, via optimizing F1 during CV depth selection rather than directly reweighting misclassifications), lands on the opposite side of the same trade-off.

## Which model would an operator prefer, and why

Depends entirely on which error costs more in context, not on which model has the higher F1:

- **If missing a real anomaly is the costly failure** (an undiagnosed spacecraft fault that could compound before the next telemetry review, say), **Logistic Regression's higher recall (0.74)** is the better choice despite its lower F1 and worse precision. Accepting more false alarms is an acceptable price for catching more real faults.
- **If operator attention is the scarce resource** (each flagged fragment requires a human to review it, and false alarms erode trust in the system or waste limited attention), **the Decision Tree's higher precision (0.77)** is preferable. It flags less often, but it's more often right when it does.

Neither model is unconditionally "better" here. The comparison demonstrates that model choice under class imbalance is an operational decision about which failure mode is more tolerable, not just a matter of picking the higher aggregate score.

## Precision-Recall curves (diagnostic only)

`figures/pr_curves.png`: Decision Tree average precision 0.76 versus Logistic Regression's 0.66, with the tree's curve sitting above Logistic Regression's for nearly the entire recall range. This nuances the point above. While Logistic Regression's specific *default* operating point (the 0.5 probability threshold both models use out of the box) favors recall over the tree's default, the tree is actually the stronger model across essentially the *whole* threshold spectrum, not just at that one point. Reported for its diagnostic value only. Per the Design Rule, no threshold gets selected or changed for either model as a result, that decision stays explicitly out of scope for this project.

## Misclassification autopsy

Per-channel plots (`figures/autopsy_{model}_{channel}_missed_vs_caught.png`) of missed anomalies (false negatives) alongside correctly-caught anomalies (true positives), reusing `run_eda`'s fragment-comparison plotting code with a new grouping.

The missed anomalies are consistently the *subtler* ones. On channels like CADC0872 and CADC0873, caught fragments show dramatic vertical drops, flatline segments, and extreme spikes, the same "sharp interruption cut into a smooth curve" signature identified back in V2, while missed fragments are comparatively smooth with no such discontinuity. Most tellingly, on CADC0888 a missed fragment has nearly the *same shape* as the caught fragments, just slightly lower amplitude and shifted later in time. Not glitchy at all, just a mildly dampened, timing-shifted version of what the model otherwise catches easily.

This lines up directly with what the feature importance ranking (V4 notes) already implied. The Decision Tree's top features, `std`, `duplicate_ratio`, `diff_std`, all measure *within-fragment spread or jumpiness*. They're well-suited to catching dramatic spikes and dropouts, which show up as high local variance and large sample-to-sample differences, but they have no mechanism for detecting a fragment that's merely somewhat lower-amplitude or slightly time-shifted relative to expectation. Nothing about that kind of subtle, shape- or timing-level deviation registers as unusual dispersion or repetition within the fragment itself. The summary-statistic feature set this project was scoped to (per the V3 Design Rule) is fundamentally better matched to glitch-like anomalies than to gentler, shape-based ones, a real limit of the feature family, not a tuning problem either model could fix.
