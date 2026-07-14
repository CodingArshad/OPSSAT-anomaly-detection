# Turns each fragment's value array into one row of numbers a model can actually take in
import numpy as np
import pandas as pd
import os

def extract_features(fragments_df):
    rows = []
    for i, fragment in fragments_df.iterrows():
        row = {
            'fragment_id' : fragment['fragment_id'],
            'mean' : np.mean(fragment['values']),
            # How spread out the values are
            'std' : np.std(fragment['values']),
            'min' : np.min(fragment['values']),
            'max' : np.max(fragment['values']),
            'range' : np.max(fragment['values']) - np.min(fragment['values']),
            # How long the fragment is
            'n_samples' : len(fragment['values']),
            'n_unique' : len(np.unique(fragment['values'])),
            # Stuck sensor = lots of repeats
            'duplicate_ratio' : 1 - (len(np.unique(fragment['values'])) / len(fragment['values'])),
            # Big sample-to-sample jumps show up here
            'diff_mean_abs' : np.mean(np.abs(np.diff(fragment['values']))),
            'diff_std' : np.std(np.diff(fragment['values'])),
            'label' : fragment['label'],
            'channel' : fragment['channel']
        }
        rows.append(row)
    features_df = pd.DataFrame(rows)

    # Turn channel string into a number
    features_df['channel'] = features_df['channel'].astype('category').cat.codes

    # Make sure nothing is wrong
    assert len(features_df) == len(fragments_df), 'Expected same number of rows in features_df as fragments_df'
    assert features_df.isnull().sum().sum() == 0, 'Expected no NaN values in features_df'

    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv('data/processed/features.csv', index=False)

    return features_df
