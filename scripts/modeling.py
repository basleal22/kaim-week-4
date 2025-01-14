import os
import pickle
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


def save_with_pickle(model, name, directory='models'):
    """Serialize the model with pickle and save it to the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'{directory}/{name}_{timestamp}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f'Model saved as {filename}')
    return filename


def preprocess_data(train_data, test_data):
    """
    Preprocess training and test data by imputing missing values
    and encoding categorical variables.
    """
    # Identify the target and features
    target = 'Sales'
    drop_columns = ['Sales', 'Customers']  # Columns to drop from training data
    id_column = 'Id'  # Column to drop from test data
    
    # Ensure target exists in training data
    if target not in train_data.columns:
        raise ValueError(f"The training data must contain the target column '{target}'.")
    if id_column not in test_data.columns:
        raise ValueError(f"The test data must contain the identifier column '{id_column}'.")
    
    # Drop unnecessary columns
    X_train = train_data.drop(columns=drop_columns, errors='ignore')
    y_train = train_data[target]
    X_test = test_data.drop(columns=[id_column], errors='ignore')

    # Ensure columns match between train and test
    common_columns = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_columns]
    X_test = X_test[common_columns]

    # Convert categorical columns to strings to avoid mixed types (int, str)
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Imputation and encoding
    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    encoder_cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Apply transformations
    X_train[numerical_cols] = imputer_num.fit_transform(X_train[numerical_cols])
    X_train[categorical_cols] = imputer_cat.fit_transform(X_train[categorical_cols])
    X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])
    X_test[categorical_cols] = imputer_cat.transform(X_test[categorical_cols])

    # Encode categorical columns
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # Align columns between train and test
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, y_train, X_test



def train_models(train_data, test_data):
    """
    Train multiple models and evaluate them using training data.
    Returns predictions on the test data, evaluation metrics, and saved model files.
    """
    try:
        X_train, y_train, X_test = preprocess_data(train_data, test_data)
    except ValueError as e:
        print(f"Error in preprocessing data: {e}")
        return None

    models = {
        'RandomForestRegressor': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror')
    }

    results = {}
    model_files = {}
    test_predictions = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        # Save the model
        filename = save_with_pickle(model, model_name)
        model_files[model_name] = filename

        # Evaluate the model
        train_predictions = model.predict(X_train)
        mse = mean_absolute_error(y_train, train_predictions)
        r2 = r2_score(y_train, train_predictions)
        results[model_name] = {'Mean Absolute Error': mse, 'R2 Score': r2}

        # Predict on test data
        test_predictions[model_name] = model.predict(X_test)

        print(f"{model_name} training complete. Metrics: MAE = {mse:.2f}, R2 = {r2:.2f}")

    return test_predictions, results, model_files
