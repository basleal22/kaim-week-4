from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
def preprocess(train_data,test_data):
    # Convert 'Date' to datetime and set it as the index
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    train_data.set_index('Date', inplace=True)
    test_data.set_index('Date', inplace=True)

    # Sort by date to maintain chronological order
    train_data.sort_index(inplace=True)
    test_data.sort_index(inplace=True)

    # Drop unnecessary columns
    X_train = train_data.drop(columns=['Sales', 'Customers'], errors='ignore')
    y_train = train_data['Sales']
    X_test = test_data.drop(columns=['Id'], errors='ignore')

    # Convert categorical columns to strings to avoid mixed types (int, str)
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Imputation
    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    scaler = StandardScaler()

    # Impute numerical columns
    X_train[numerical_cols] = imputer_num.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])

    # Impute categorical columns
    X_train[categorical_cols] = imputer_cat.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = imputer_cat.transform(X_test[categorical_cols])

    # Encode categorical columns using one-hot encoding
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # Align the test set with the training set columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Apply standard scaling to numerical columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Resample the data for time series analysis (e.g., daily aggregation)
    X_train = X_train.resample('D').mean()
    y_train = y_train.resample('D').sum()  # Sum sales for each day
    X_test = X_test.resample('D').mean()

    # Fill missing values created by resampling
    X_train.fillna(method='ffill', inplace=True)
    y_train.fillna(method='ffill', inplace=True)
    X_test.fillna(method='ffill', inplace=True)

    return X_train, y_train, X_test
def create_supervised_data(data, n_lags=3):
    """
    Transform a time series dataset into supervised learning data.
    
    Parameters:
    - data (pd.Series): The target time series (e.g., Sales).
    - n_lags (int): Number of lag observations to use as features.

    Returns:
    - X (np.ndarray): Features for supervised learning.
    - y (np.ndarray): Target values for supervised learning.
    """
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i - n_lags:i].values)
        y.append(data[i])
    return np.array(X), np.array(y)
def scale_data(X, y):
    """
    Scale features and target using MinMaxScaler to range (-1, 1).
    
    Parameters:
    - X (np.ndarray): Features for supervised learning.
    - y (np.ndarray): Target values for supervised learning.

    Returns:
    - X_scaled (np.ndarray): Scaled features.
    - y_scaled (np.ndarray): Scaled target.
    - scaler (MinMaxScaler): Fitted scaler for inverse transformations.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    return X_scaled, y_scaled, scaler

def build_lstm_model(input_shape):
    """
    Build an LSTM model for regression tasks.

    Parameters:
    - input_shape (tuple): Shape of the input data (time steps, features).

    Returns:
    - model (Sequential): Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the LSTM model
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Train the LSTM model.

    Parameters:
    - model (Sequential): Compiled LSTM model.
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.

    Returns:
    - history: Training history object.
    """
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the trained model and calculate RMSE.

    Parameters:
    - model (Sequential): Trained LSTM model.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): Test target.
    - scaler (MinMaxScaler): Fitted scaler for inverse transformations.

    Returns:
    - rmse (float): Root Mean Squared Error.
    - y_pred (np.ndarray): Predicted values (inverse scaled).
    - y_actual (np.ndarray): Actual values (inverse scaled).
    """
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    return rmse, y_pred, y_actual


    