# Goes through fragments_df and drops anything that looks broken
# (duplicate ids, empty fragments, NaNs). Also prints how much got dropped
# so there's a record of it instead of silently deleting rows.
import pandas as pd
import numpy as np

def clean_fragments(dataset):
    dataset = dataset.copy()  # don't want to modify the df someone passed in

    # if the same fragment_id shows up twice, something's wrong with the source
    # data, just keep the first one and drop the rest
    duplicates = dataset['fragment_id'].duplicated().sum()
    print(f'There are {duplicates} duplicate fragment IDs in the dataset.')
    dataset = dataset[~dataset['fragment_id'].duplicated()]

    # a fragment with 0 samples in it isn't useful for anything, can't compute
    # stats on an empty array
    empty = dataset['values'].apply(lambda x: len(x) == 0).sum()
    print(f'There are {empty} empty fragments in the dataset.')
    dataset = dataset[dataset['values'].apply(lambda x: len(x) > 0)]

    # checking both "is it even numbers" and "does it have NaN in it" at once
    # since either one would break the feature math later
    def is_bad(x):
        if not np.issubdtype(x.dtype, np.number):
            return True
        return np.isnan(x).any()

    nan = dataset['values'].apply(is_bad).sum()
    print(f'There are {nan} fragments with NaN values in the dataset.')
    dataset = dataset[~dataset['values'].apply(is_bad)]

    # reset the index since we dropped rows above and the old index would
    # have gaps in it now
    dataset.reset_index(drop=True, inplace=True)
    print(f'After cleaning, there are {len(dataset)} fragments remaining in the dataset.')
    return dataset
