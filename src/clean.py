# Drops fragments that look broken (dupes, empty, NaNs) and logs how much got dropped
import pandas as pd
import numpy as np

def clean_fragments(dataset):
    # Don't edit original data
    dataset = dataset.copy()

    # Check for duplicate data
    duplicates = dataset['fragment_id'].duplicated().sum()
    print(f'There are {duplicates} duplicate fragment IDs in the dataset.')
    dataset = dataset[~dataset['fragment_id'].duplicated()]

    # Check for empty data
    empty = dataset['values'].apply(lambda x: len(x) == 0).sum()
    print(f'There are {empty} empty fragments in the dataset.')
    dataset = dataset[dataset['values'].apply(lambda x: len(x) > 0)]

    # Check for NaN data
    def is_bad(x):
        if not np.issubdtype(x.dtype, np.number):
            return True
        return np.isnan(x).any()

    nan = dataset['values'].apply(is_bad).sum()
    print(f'There are {nan} fragments with NaN values in the dataset.')
    dataset = dataset[~dataset['values'].apply(is_bad)]

    # Old index now has gaps
    dataset.reset_index(drop=True, inplace=True)
    print(f'After cleaning, there are {len(dataset)} fragments remaining in the dataset.')
    return dataset
