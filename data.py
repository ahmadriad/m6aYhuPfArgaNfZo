import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sb
from sklearn.model_selection import GridSearchCV
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("C:/Users/97155/Downloads/term-deposit-marketing-2020.csv")

def split_data(data):
    label_encoder = LabelEncoder()
    data["y"]=label_encoder.fit_transform(data["y"])
    encoder = ce.BinaryEncoder(cols=categorical_columns)
    Binary_encoded_data = encoder.fit_transform(data)
    minority_samples = Binary_encoded_data[Binary_encoded_data['y'] == 1]
    num_minority_samples = len(minority_samples)
    majority_samples = Binary_encoded_data[Binary_encoded_data['y'] == 0].sample(n=num_minority_samples, random_state=seed)
    undersampled_data = pd.concat([minority_samples, majority_samples])
    undersampled_features = undersampled_data.drop('y', axis=1)
    undersampled_target = undersampled_data['y']
    undersampled_target=pd.DataFrame(undersampled_target)
    return undersampled_features,undersampled_target
