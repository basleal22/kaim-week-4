import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def extract_data(data):
    csv_data=pd.read_csv(data)
    return csv_data
def Data_cleaning(data):
    numerical_cols=data.select_dtypes(include=['float64','int64'])
    categorical_cols=data.select_dtypes(include=['object','category'])
    for column in data.columns:
        if column in numerical_cols.columns:
            data[column] = data[column].fillna(data[column].mean())
        elif column in categorical_cols.columns:
            data[column] = data[column].fillna(data[column].mode().iloc[0])
    return data
def outliers(data):
    numerical_cols=data.select_dtypes(include=['float64','int64'])
    q1=numerical_cols.quantile(0.25)
    q3=numerical_cols.quantile(0.75)
    IQR=q3-q1
    lower_bound=q1-1.5*IQR
    higher_bound=q3+1.5*IQR
    data_mean=numerical_cols.mean()
    for column in numerical_cols.columns:
        mask = (numerical_cols[column] < lower_bound[column]) | (numerical_cols[column] > higher_bound[column])
        data.loc[mask, column] = data_mean[column]
    return data