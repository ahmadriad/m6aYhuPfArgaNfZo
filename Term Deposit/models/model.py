
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def train_model(undersampled_features, undersampled_target):
    seed=1
    np.random.seed(seed)
    clf_xg = XGBClassifier(max_depth=10,random_state=seed,reg_lambda=10)
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    scores_xg = cross_val_score(clf_xg, undersampled_features, undersampled_target, cv=kf, scoring='f1_macro')
    
    return scores_xg


def get_model_metrics(model):
    for i, score in enumerate(model):
        print(f"Fold {i+1}: f1_score = {score:.4f}")
    average_accuracy_xg = model.mean()
    print(f"Average f1_score: {average_accuracy_xg:.4f}")
