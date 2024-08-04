# backend.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
import statsmodels.api as sm

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data['START_DATE'] = pd.to_datetime(data['START_DATE'], errors='coerce')
    data['END_DATE'] = pd.to_datetime(data['END_DATE'], errors='coerce')
    data['DATE'] = pd.DatetimeIndex(data['START_DATE']).date
    data['TIME'] = pd.DatetimeIndex(data['START_DATE']).hour
    data['SHIFT'] = pd.cut(data['TIME'], bins=[0, 10, 15, 19, 24], labels=['Morning', 'Afternoon', 'Evening', 'Night'])
    data['ETA'] = (data['END_DATE'] - data['START_DATE']).dt.total_seconds() / 60
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def encode_data(data):
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(data[['CATEGORY', 'PURPOSE']]))
    OH_cols.index = data.index
    OH_cols.columns = OH_encoder.get_feature_names_out(['CATEGORY', 'PURPOSE'])
    df_final = data.drop(['CATEGORY', 'PURPOSE'], axis=1)
    data_encoded = pd.concat([df_final, OH_cols], axis=1)
    return data_encoded

def encode_data2(data):
    obj = (data.dtypes == 'object') | (data.dtypes == 'category')
    object_cols = list(obj[obj].index)
    label_encoders = {}

    for column in object_cols:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

    return data

def split_data(data):
    X = data.iloc[:, 2:-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def load_model(file_path):
    return models.load_model(file_path)

def preprocess_input(new_data, scaler):
    return scaler.transform(new_data)

def make_prediction(model, new_data_scaled):
    predictions = model.predict(new_data_scaled)
    return predictions

# Main function to run the entire backend process
def main():
    file_path = "UberDataset.csv"
    data = load_data(file_path)
    data = preprocess_data(data)
    data = encode_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    return scaler

if __name__ == "__main__":
    scaler = main()
