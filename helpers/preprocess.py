# helpers/preprocess.py

import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def handle_missing_values(df, method='drop', fill_value=None):
    if method == 'drop':
        df_cleaned = df.dropna()
    elif method == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif method == 'fill':
        df_cleaned = df.fillna(fill_value)
    else:
        df_cleaned = df.copy()
    return df_cleaned

def encode_categoricals(df, method='label'):
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns

    if method == 'label':
        for col in cat_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

    return df_encoded

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def scale_numerical(df, method='standard'):
    df_scaled = df.copy()

    # Select numeric columns only
    num_cols = df_scaled.select_dtypes(include=['number']).columns.tolist()

    if not num_cols:
        return df_scaled

    df_scaled[num_cols] = df_scaled[num_cols].apply(pd.to_numeric, errors='coerce')
    df_scaled[num_cols] = df_scaled[num_cols].fillna(0)

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        return df_scaled

    try:
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    except Exception as e:
        print("Scaling Error:", e)

    return df_scaled

