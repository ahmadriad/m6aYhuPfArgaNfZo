
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

data = pd.read_csv("C:/Users/97155/Downloads/term-deposit-marketing-2020.csv")


label_encoder = LabelEncoder()
columns_to_encode = [1, 2, 3, 4, 6, 7, 8,10,13]

for column in columns_to_encode:
    data.iloc[:, column] = label_encoder.fit_transform(data.iloc[:, column])

features = data.drop(["y"],axis=1)
target=data["y"]


scl = MinMaxScaler(feature_range=(0, 1))
features_scl = scl.fit_transform(features)
features_scl = pd.DataFrame(features_scl, columns=features.columns)

np.random.seed(seed)

clf_xg = XGBClassifier(max_depth=10,random_state=seed,reg_lambda=10)


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=1)
scores_xg = cross_val_score(clf_xg, features, target, cv=kf, scoring='accuracy')
for i, score in enumerate(scores_xg):
    print(f"Fold {i+1}: Accuracy = {score:.4f}")
average_accuracy_xg = scores_xg.mean()
print(f"Average Accuracy: {average_accuracy_xg:.4f}")


target_one_indices = target[target == 1].index
duplicated_features = pd.concat([features, features.loc[target_one_indices]], ignore_index=True)
duplicated_target = pd.concat([target, target.loc[target_one_indices]], ignore_index=True)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=1)
scores_xg = cross_val_score(clf_xg, duplicated_features, duplicated_target, cv=kf, scoring='accuracy')
for i, score in enumerate(scores_xg):
    print(f"Fold {i+1}: Accuracy = {score:.4f}")
average_accuracy_xg = scores_xg.mean()
print(f"Average Accuracy: {average_accuracy_xg:.4f}")
