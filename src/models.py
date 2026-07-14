# Trains both models and saves them so I don't have to retrain every time
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_models(X_train, y_train):
    # Keeping this small on purpose, not an open search
    depths = [3, 4, 5, 6]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    best_depth = None
    best_score = -1

    # 5-fold CV on training data only, test set stays untouched
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=1)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        mean_score = scores.mean()

        print(f'max_depth={depth}: mean CV F1 = {mean_score:.3f}')

        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    # Refit on all training data now
    refitted = DecisionTreeClassifier(max_depth=best_depth, random_state=1).fit(X_train, y_train)
    os.makedirs('results/models/', exist_ok=True)
    joblib.dump(refitted, 'results/models/decision_tree.joblib')

    # One-hot, tree used ordinal codes but LR shouldn't
    X_train_encoded = pd.get_dummies(X_train, columns=['channel'])
    log_reg = Pipeline([
        # LR needs same-scale features, tree doesn't care
        ('scaler', StandardScaler()),
        # Balanced = weigh the minority class more
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1))
    ])
    log_reg.fit(X_train_encoded, y_train)
    joblib.dump(log_reg, 'results/models/logistic_regression.joblib')

    return {'decision_tree': refitted, 'logistic_regression': log_reg}
