# helpers/eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot theme
sns.set_style("whitegrid")
    
def show_basic_info(df):
    st.subheader("📋 Descriptive Statistics")
    st.write(df.describe(include='all'))

def show_missing_values(df):
    st.subheader("🧩 Missing Value Heatmap")
    plt.figure(figsize=(10, 2))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    st.pyplot(plt.gcf())
    plt.clf()

def show_correlation_heatmap(df):
    st.subheader("📈 Correlation Heatmap (Numerical)")
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    plt.figure(figsize=(10,2))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt.gcf())
    plt.clf()
    
def show_distribution(df):
    st.subheader("📊 Distribution Plots (Numerical)")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        st.markdown(f"**{col}**")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

def show_categorical_counts(df):
    st.subheader("🔠 Count Plots (Categorical)")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        st.markdown(f"**{col}**")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close(fig)
