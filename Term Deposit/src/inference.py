import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import numpy as np
from xgboost import XGBClassifier

def preprocess_data(input_data):
    label_encoder = LabelEncoder()
    input_data["y"] = label_encoder.fit_transform(input_data["y"])
    categorical_columns = input_data.select_dtypes(include=['object']).columns
    encoder = ce.BinaryEncoder(cols=categorical_columns)
    preprocessed_data = encoder.fit_transform(input_data)
    return preprocessed_data

def load_model(model_path):
    return joblib.load(model_path)

def perform_inference(model, input_data):
    # Preprocess the input data
    preprocessed_data = preprocess_data(input_data)
    # Extract the features
    inference_features = preprocessed_data.drop('y', axis=1)

    # Perform inference using the loaded model
    predictions = model.predict(inference_features)
    return predictions

def main():
    # Load the trained model
    model_path = "XGBoost_model.pkl"
    model = load_model(model_path)

    # Load the input data for inference
    input_data = pd.read_csv("C:/Users/97155/Downloads/term-deposit-marketing-2020.csv")

    # Perform inference using the loaded model
    predictions = perform_inference(model, input_data)

    # Save the predictions to a file
    np.savetxt("predictions.csv", predictions, delimiter=",")

    print("Inference completed successfully. Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()
