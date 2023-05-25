import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sb
from sklearn.model_selection import GridSearchCV
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def main():
    data = pd.read_csv("C:/Users/97155/Downloads/term-deposit-marketing-2020.csv")

    data = split_data(data)

  
    xgb_model = train_model(undersampled_features, undersampled_target)

    metrics = get_model_metrics(model)

    model_name = "XGBoost_model.pkl"

    joblib.dump(value=xgb_model, filename=model_name)

if __name__ == "__main__":
    main()
