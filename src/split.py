# Splits features_df into a training set and a test set. Doing this before
# any model exists so no model choice can accidentally influence the split.
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(features_df):
    # dropping fragment_id since it's just an id, not something to learn from
    # dropping label since that's y, not X
    X = features_df.drop(columns=['fragment_id', 'label'])
    y = features_df['label']

    # stratify=y matters a lot here since only ~20% of rows are anomalous,
    # without it a random split could end up with barely any anomalies in
    # the test set which would make recall basically meaningless
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    train_anomaly = y_train.mean()
    test_anomaly = y_test.mean()
    overall_rate = y.mean()

    print(f'The training set has {len(X_train)} samples with an anomaly rate of {train_anomaly:.2%}')
    print(f'The test set has {len(X_test)} samples with an anomaly rate of {test_anomaly:.2%}')
    print(f'The overall set has an anomaly rate of {overall_rate:.2%}')

    # double checking that stratify actually did its job, if these fail the
    # split is probably broken somehow
    assert abs(train_anomaly - overall_rate) < 0.02, 'Anomaly rates between training and testing sets differ by more than 2%'
    assert abs(test_anomaly - overall_rate) < 0.02, 'Anomaly rates between training and testing sets differ by more than 2%'

    return X_train, X_test, y_train, y_test
