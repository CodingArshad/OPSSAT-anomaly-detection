# Trains both models and saves them so I don't have to retrain every time
# I want to look at the results again.
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_models(X_train, y_train):
    # keeping this range small on purpose, not trying every possible depth,
    # just picking the best one out of a few reasonable options
    depths = [3, 4, 5, 6]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    best_depth = None
    best_score = -1

    # trying each depth with 5-fold cross validation on the TRAINING set only,
    # the test set doesn't get touched until evaluate_models later
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=1)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        mean_score = scores.mean()

        print(f'max_depth={depth}: mean CV F1 = {mean_score:.3f}')

        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    # now that I know the best depth, train one final tree on all the
    # training data (not just the cv folds) using that depth
    refitted = DecisionTreeClassifier(max_depth=best_depth, random_state=1).fit(X_train, y_train)
    os.makedirs('results/models/', exist_ok=True)
    joblib.dump(refitted, 'results/models/decision_tree.joblib')

    # logistic regression needs channel one-hot encoded instead of the
    # ordinal 0-8 codes from extract_features, a linear model would
    # otherwise assume channel 8 is somehow "more" than channel 0
    X_train_encoded = pd.get_dummies(X_train, columns=['channel'])
    log_reg = Pipeline([
        ('scaler', StandardScaler()),  # LR needs features on the same scale, tree doesn't
        # class_weight='balanced' tells sklearn to care more about getting the
        # anomalous class right, since it's the minority class here
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1))
    ])
    log_reg.fit(X_train_encoded, y_train)
    joblib.dump(log_reg, 'results/models/logistic_regression.joblib')

    return {'decision_tree': refitted, 'logistic_regression': log_reg}
