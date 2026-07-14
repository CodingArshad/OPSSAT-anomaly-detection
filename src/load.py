# Loads the raw OPSSAT-AD files (dataset.csv + segments.csv) and turns them
# into one DataFrame where each row is a full fragment (not a single sample).
# Spec for this is in docs/algorithm.md under V1 if I forget the details later.
import pandas as pd
import numpy as np

def load_fragments(data_dir='data/raw'):
    # dataset.csv has one row per fragment (label, channel, etc)
    # segments.csv has one row per individual sample, way more rows
    dataset = pd.read_csv(data_dir + '/dataset.csv')
    segments = pd.read_csv(data_dir + '/segments.csv')

    # samples need to be in time order before I turn them into arrays,
    # otherwise the values array would be scrambled
    sorted_segments = segments.sort_values('timestamp')
    grouped = sorted_segments.groupby('segment')

    # this squishes every fragment's samples into one numpy array per fragment
    # so instead of many rows we get a single 'values' column of arrays
    values = grouped['value'].apply(lambda group: group.to_numpy())
    timestamps = grouped['timestamp'].apply(lambda group: group.to_numpy())

    # the original column names don't match what I'm using everywhere else,
    # so renaming them once here instead of doing this in every file
    dataset.rename(columns={'segment': 'fragment_id', 'anomaly': 'label'}, inplace=True)

    # dataset.csv has some extra columns I don't need, just keeping what matters
    dataset = dataset[['fragment_id', 'label', 'channel']]

    # setting fragment_id as the index so the values/timestamps line up with
    # the right row when I assign them, then undoing it so fragment_id is
    # back to being a normal column
    dataset.set_index('fragment_id', inplace=True)
    dataset['values'] = values
    dataset['timestamps'] = timestamps
    dataset.reset_index(inplace=True)

    # these numbers are published facts about this dataset (from the paper),
    # so if any of these fail something went wrong in the steps above
    assert len(dataset) == 2123, 'Expected 2123 fragments'
    assert dataset['channel'].nunique() == 9, 'Expected 9 channels'
    assert dataset['label'].mean() >= 0.15 and dataset['label'].mean() <= 0.25, 'Expected anomaly rate between 15% and 25%'
    print(f'Loaded {len(dataset)} fragments with {dataset["channel"].nunique()} channels and an anomaly rate of {dataset["label"].mean():.2%}')

    return dataset
