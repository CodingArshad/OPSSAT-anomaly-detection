"""clean_fragments: audit fragments_df for data-quality issues and log every decision.

See docs/algorithm.md (V1) for the spec: what to check for (duplicate
fragment_ids, empty fragments, NaN/non-numeric values) and what to log.
"""
import pandas as pd
import numpy as np

def clean_fragments(dataset):
    dataset = dataset.copy()

    # Check for duplicate fragment IDs
    duplicates = dataset['fragment_id'].duplicated().sum()
    print(f'There are {duplicates} duplicate fragment IDs in the dataset.')
    dataset = dataset[~dataset['fragment_id'].duplicated()]

    # Check for empty fragments
    empty = dataset['values'].apply(lambda x: len(x) == 0).sum()
    print(f'There are {empty} empty fragments in the dataset.')
    dataset = dataset[dataset['values'].apply(lambda x: len(x) > 0)]

    # Check for NaN or non-numeric values
    def is_bad(x):
        if not np.issubdtype(x.dtype, np.number):
            return True
        return np.isnan(x).any()
        
    nan = dataset['values'].apply(is_bad).sum()
    print(f'There are {nan} fragments with NaN values in the dataset.')
    dataset = dataset[~dataset['values'].apply(is_bad)]

    # Final bit
    dataset.reset_index(drop=True, inplace=True)
    print(f'After cleaning, there are {len(dataset)} fragments remaining in the dataset.')
    return dataset