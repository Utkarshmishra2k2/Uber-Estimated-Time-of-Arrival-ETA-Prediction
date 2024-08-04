# model_file.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.callbacks import LearningRateScheduler
data_01 = pd.read_csv("https://raw.githubusercontent.com/UM1412/Data-Set/main/UberDataset.csv")
data_01['PURPOSE'].fillna("Unkonwn", inplace=True)
data_01['START_DATE'] = pd.to_datetime(data_01['START_DATE'],errors='coerce')
data_01['END_DATE'] = pd.to_datetime(data_01['END_DATE'],errors='coerce')
data_01['DATE'] = pd.DatetimeIndex(data_01['START_DATE']).date
data_01['TIME'] = pd.DatetimeIndex(data_01['START_DATE']).hour
data_01['SHIFT'] = pd.cut(x=data_01['TIME'],bins = [0,10,15,19,24],labels = ['Morning','Afternoon','Evening','Night'])
data_01['ETA'] = (data_01['END_DATE'] - data_01['START_DATE']).dt.total_seconds() / 60
data_01.dropna(inplace=True)
data_01.drop_duplicates(inplace=True)
obj = (data_01.dtypes == 'object') | (data_01.dtypes == 'category')
object_cols = list(obj[obj].index)
label_encoders = {}
for column in object_cols:
    le = LabelEncoder()
    data_01[column] = le.fit_transform(data_01[column].astype(str))
    label_encoders[column] = le
X = data_01.iloc[:,2:-1]
y = data_01.iloc[:,-1]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * (2.71828 ** -0.1))

callback = LearningRateScheduler(scheduler)
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),  # Increased number of neurons
    layers.Dropout(0.3),  # Added dropout
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Added dropout
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])
# Print model summary
print(model.summary())
model.compile(optimizer='adam', loss='mean_absolute_error')
history = model.fit(X_train_scaled, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[callback])
model.save('deep_eta_model.h5')
