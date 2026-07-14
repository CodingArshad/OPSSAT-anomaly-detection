# V2 EDA Observations

Notes on `channel_counts_anomaly_rates.png` and the 9 `fragments_{channel}_anomalous_vs_nominal.png` figures.

## Cross-channel pattern

Nominal fragments trace smooth, roughly bell-shaped curves, likely tracking something physically continuous, maybe the satellite's orientation relative to the sun or Earth cycling once per orbit.

Anomalous fragments largely follow that same underlying smooth shape but have sharp, localized interruptions cut into it: sudden spikes, vertical level shifts, or flatlines that break an otherwise continuous curve before it resumes. Shows up across nearly every channel (clearest in CADC0873, CADC0888, CADC0890) and is the main visual signature of an anomaly here. Not a totally different shape. A smooth signal with a glitch layered on top.

This suggests V3 features sensitive to *local* discontinuity (large sample-to-sample differences, second-derivative magnitude, peak/spike counts) are more likely to separate the classes than global summary stats like plain mean or variance, which would wash out a brief, sharp interruption.

## Channel-specific notes

- **CADC0886**: nominal fragments peak around 0.6-0.7, anomalous fragments cap out around 0.25-0.4. A visible amplitude reduction, not just a shape distortion. Anomalies here look "damped" relative to nominal behavior.
- **CADC0872**: fragments mostly overlap in duration (roughly 0-200s elapsed), but one sampled anomalous fragment runs much longer (~900s) than any other fragment shown. Worth remembering that fragment duration itself varies a lot (8-1,040 samples per the raw archive) and isn't controlled for in these plots.

## Anomaly rate is not uniform across channels

Per `channel_counts_anomaly_rates.png`, rates range from 0.00% (CADC0884) to 78.57% (CADC0890), most channels landing in the 15-35% range. CADC0886 and CADC0890 have very few total fragments (11 and 14 respectively), so those extreme rates need a small-sample caveat attached. A couple of mislabeled or unusual fragments would swing the percentage a lot more there than on a channel with a large fragment count.

---

# V3 EDA Observations (pass 2, features)

Notes on the 10 `feature_{name}_anomalous_vs_nominal.png` histograms (`features_df`, split by label, density-normalized with shared bin edges across both classes).

## Strongest separators

- **`diff_mean_abs`**: the clearest split of any feature. Nominal fragments concentrate heavily right at 0 (small sample-to-sample changes throughout), anomalous fragments have a visibly longer tail into higher values. Direct, quantitative confirmation of the V2 finding above, the "sharp interruption cut into a smooth curve" pattern shows up numerically as larger first-differences, exactly as predicted before any feature was even computed.
- **`n_samples` / `n_unique`**: this one was unexpected, not predicted from the V2 plots at all. Nominal fragments concentrate heavily at short lengths (mostly under 100 samples), anomalous fragments spread out much further, extending past 400-600 samples with a secondary cluster around 150-250. Fragment length itself correlates with anomaly status, worth keeping in mind for V4 since a model could partly be learning "is this fragment unusually long" rather than anything about the actual shape of the signal.

## Moderate / weak separators

- **`diff_std`**: some separation (anomalous has a taller first bin, more density at very small variability) but less dramatic than `diff_mean_abs`.
- **`max`, `range`**: mostly overlapping, nominal shows a slightly fuller tail at higher values. Moderate at best.
- **`std`, `min`, `duplicate_ratio`**: heavy overlap between classes across their full range. These look like the weakest of the 10, plausibly contributing little on their own once a model is trained, though a tree-based model could still find them useful in combination with other features.

## Known confound: `mean` (and any raw-scale feature) mixes channels with very different physical scales

`mean`'s histogram shows a real shape difference between classes, but it's pooling all 9 channels together in one chart, and channels differ by orders of magnitude in raw scale (CADC0872 values sit around 1e-5, while CADC0886/CADC0890 range roughly 0-1, per the V2 fragment plots). A single cross-channel histogram can't distinguish "this differs because of anomaly status" from "this differs because of which channel it came from." Not a bug in `extract_features` or `run_eda`, the spec only calls for one overlaid histogram per feature, not a per-channel breakdown, but a real limitation to remember when interpreting `mean` (and to a lesser extent `min`/`max`/`range`, which share the same scale issue).

## A methodology note for future EDA: mismatched histogram bins can fake a signal

Early versions of these plots used independent, unspecified bin edges for the anomalous and nominal `plt.hist()` calls. Since Matplotlib auto-computes bins per call based on that call's own data range, and the two classes have different spreads, this produced histograms that looked like they had a "big gap that closes further out," which turned out to be largely a binning artifact, not real signal. Fixed by computing one shared `bins` array (via `np.linspace` over the full column's min/max) and passing the same `bins=` to both `plt.hist()` calls. Worth remembering for any future overlaid-histogram comparison: mismatched bins between two overlaid histograms aren't a valid apples-to-apples comparison.

## Net takeaway

Pass-2 EDA confirms the spec's core question. Yes, several features visibly separate the classes (`diff_mean_abs` clearly, `n_samples`/`n_unique` clearly if unexpectedly, `diff_std` moderately), while several others carry weak or ambiguous signal alone (`std`, `min`, `duplicate_ratio`). Reasonable to proceed to `split_data` and V4 modeling on this feature set as-is, per the V3 Design Rule. No feature list changes are justified by anything found here, since every original feature was already tied to a V2 observation and none turned out to be *completely* uninformative.

---

# V4 Observations (Decision Tree)

## Cross-validation depth selection

5-fold stratified CV on the training set only, comparing mean F1 across `max_depth ∈ {3,4,5,6}`:

| max_depth | mean CV F1 |
|-----------|------------|
| 3         | 0.685      |
| 4         | 0.771      |
| 5         | 0.804      |
| 6         | 0.799      |

Depth 5 won. The shape is exactly what the Design Rule predicts: performance climbs as the tree gets more expressive, peaks at 5, then dips very slightly at 6, a hint of overfitting starting to creep in once the tree is deep enough that 1,698 training rows stop fully supporting it.

## Test-set performance (evaluated once, per the frozen-split rule)

Anomalous class (positive): precision 0.77, recall 0.66, F1 0.71. Confusion matrix: 321 true nominal, 17 false alarms, 30 missed anomalies, 57 correctly caught anomalies. Accuracy 88.94% against a 79.53% always-predict-nominal baseline (footnote only, per the Design Rule, the real story is that roughly 1 in 3 real anomalies, 30 of 87, still slip through as false negatives).

## Feature importance diverges sharply from the V3 univariate ranking

The tree's `.feature_importances_` ranks `std` (0.39), `duplicate_ratio` (0.26), and `diff_std` (0.18) as by far the most-relied-on features, together over 83% of total importance. `n_unique` contributes a modest 0.07. Everything else, including `diff_mean_abs`, `mean`, `channel`, `n_samples`, `min`, `max`, `range`, is marginal or effectively unused.

This is a real surprise relative to V3's pass-2 EDA, which flagged `diff_mean_abs` as the single clearest univariate separator and `std`/`duplicate_ratio` as weak, heavily-overlapping features. Likely explanation: pass-2 EDA measures each feature's separating power *in isolation*, while `.feature_importances_` measures how much a feature adds *given the other features the tree already used*. Several of the diff-based and level-based features are probably correlated with each other (a fragment with one large jump likely scores high on both `diff_mean_abs` and `diff_std`, for instance), so once the tree splits on one of a correlated group, the others contribute little *additional* separating power even if they'd have looked informative on their own. Univariate EDA and a trained multivariate model's actual reliance on features are answering genuinely different questions, and this result is a concrete example of them disagreeing.

## Reference-repo sanity check (30-min timebox)

The kplabs-pl/OPS-SAT-AD paper's own 30-algorithm benchmark reports a best F1 of 0.929 (fully-connected neural network, precision 0.963, recall 0.929), well above this tree's 0.71. But the protocols aren't very comparable: a 30-algorithm sweep including deep learning approaches, versus one depth-constrained, interpretable tree on 10 hand-built summary-statistic features, likely with different train/test methodology too. So the gap is expected, not itself a sign of a pipeline bug. Reconciling it further is out of scope per the Design Rule.
