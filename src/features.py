# Turns each fragment's raw value array into a row of numbers (a "feature
# vector"). Models can't take in a variable-length array directly, so this
# is what makes the sklearn models possible at all.
import numpy as np
import pandas as pd
import os

def extract_features(fragments_df):
    rows = []
    # looping row by row here isn't the fastest way to do this in pandas but
    # it's way easier for me to read and debug than a fully vectorized version
    for i, fragment in fragments_df.iterrows():
        row = {
            'fragment_id' : fragment['fragment_id'],
            'mean' : np.mean(fragment['values']),
            'std' : np.std(fragment['values']),   # how spread out the values are
            'min' : np.min(fragment['values']),
            'max' : np.max(fragment['values']),
            'range' : np.max(fragment['values']) - np.min(fragment['values']),
            'n_samples' : len(fragment['values']),          # how long the fragment is
            'n_unique' : len(np.unique(fragment['values'])),
            # if a sensor gets stuck it repeats the same value a bunch, this
            # feature is basically "how much repeating is going on"
            'duplicate_ratio' : 1 - (len(np.unique(fragment['values'])) / len(fragment['values'])),
            # np.diff gives sample-to-sample changes, big values here mean a
            # sudden jump or spike somewhere in the fragment
            'diff_mean_abs' : np.mean(np.abs(np.diff(fragment['values']))),
            'diff_std' : np.std(np.diff(fragment['values'])),
            'label' : fragment['label'],
            'channel' : fragment['channel']
        }
        rows.append(row)
    features_df = pd.DataFrame(rows)
    # channel is a string like "CADC0872", models need numbers, so turning it
    # into category codes (0, 1, 2...) here
    features_df['channel'] = features_df['channel'].astype('category').cat.codes

    # quick sanity checks, if either of these fail I broke something above
    assert len(features_df) == len(fragments_df), 'Expected same number of rows in features_df as fragments_df'
    assert features_df.isnull().sum().sum() == 0, 'Expected no NaN values in features_df'

    # saving this so I don't have to recompute features every time I just
    # want to look at them
    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv('data/processed/features.csv', index=False)

    return features_df
