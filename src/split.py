# Splits features_df into train/test, done before any model exists so nothing can leak in
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(features_df):
    # Id isn't a feature, label is y not X
    X = features_df.drop(columns=['fragment_id', 'label'])
    y = features_df['label']

    # Stratify matters, only ~20% anomalous
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    train_anomaly = y_train.mean()
    test_anomaly = y_test.mean()
    overall_rate = y.mean()

    print(f'The training set has {len(X_train)} samples with an anomaly rate of {train_anomaly:.2%}')
    print(f'The test set has {len(X_test)} samples with an anomaly rate of {test_anomaly:.2%}')
    print(f'The overall set has an anomaly rate of {overall_rate:.2%}')

    # Stratify should guarantee this
    assert abs(train_anomaly - overall_rate) < 0.02, 'Anomaly rates between training and testing sets differ by more than 2%'
    assert abs(test_anomaly - overall_rate) < 0.02, 'Anomaly rates between training and testing sets differ by more than 2%'

    return X_train, X_test, y_train, y_test
