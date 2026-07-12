import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train_models(X_train, y_train):
    depths = [3, 4, 5, 6]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    best_depth = None
    best_score = -1

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=1)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        mean_score = scores.mean()

        print(f'max_depth={depth}: mean CV F1 = {mean_score:.3f}')

        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    refitted = DecisionTreeClassifier(max_depth=best_depth, random_state=1).fit(X_train, y_train)
    os.makedirs('results/models/', exist_ok=True)
    joblib.dump(refitted, 'results/models/decision_tree.joblib')
    return {'decision_tree': refitted}
