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

def encode_categoricals(df, max_onehot_classes=10):
    """Smart encoding: One-hot small categoricals, LabelEncode large categoricals."""
    
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        unique_values = df_encoded[col].nunique()

        if unique_values <= max_onehot_classes:
            # One-Hot Encode
            df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
        else:
            # Label Encode after converting fully to string
            df_encoded[col] = df_encoded[col].astype(str)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded


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

