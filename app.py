import streamlit as st
import pandas as pd
import helpers.eda as eda
from helpers import preprocess

# --- Page Setup ---
st.set_page_config(page_title="No-Code ML Builder", layout="wide")
st.title("🧠 No-Code Machine Learning Builder")
st.markdown("Upload your dataset and begin your no-code ML journey!")

# --- Upload Section ---
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # --- Load Data ---
    df = pd.read_csv(uploaded_file)

    # --- Dataset Overview ---
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("🧬 Column Details")
    st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    # --- Exploratory Data Analysis ---
    st.markdown("---")
    st.header("🧪 Exploratory Data Analysis")

    with st.expander("📋 Descriptive Statistics"):
        eda.show_basic_info(df)

    with st.expander("🧩 Missing Values"):
        eda.show_missing_values(df)

    with st.expander("📈 Correlation Heatmap"):
        eda.show_correlation_heatmap(df)

    with st.expander("📊 Distribution Plots"):
        eda.show_distribution(df)

    with st.expander("🔠 Count Plots"):
        eda.show_categorical_counts(df)

    # --- Data Preprocessing ---
    st.markdown("---")
    st.header("🧼 Data Preprocessing")

    # Handle Missing Values
    st.subheader("🧩 Handle Missing Values")
    missing_option = st.selectbox("Choose missing value strategy", ['drop', 'mean', 'fill'])
    fill_val = None
    if missing_option == 'fill':
        fill_val = st.text_input("Fill missing values with:", "0")
        df = preprocess.handle_missing_values(df, method=missing_option, fill_value=fill_val)
    else:
        df = preprocess.handle_missing_values(df, method=missing_option)

    # Encode Categorical Columns
    st.subheader("🔤 Encode Categorical Columns")
    max_classes = st.slider("Max Unique Values for One-Hot Encoding", 2, 50, 10)  
    df = preprocess.encode_categoricals(df, max_onehot_classes=max_classes)

    # Scale Numerical Features
    st.subheader("📏 Scale Numerical Features")
    scale_option = st.selectbox("Choose scaling method", ['none', 'standard', 'minmax'])
    if scale_option != 'none':
        df = preprocess.scale_numerical(df, method=scale_option)

    # Show Transformed Data
    st.subheader("🧾 Transformed Data Preview")
    st.dataframe(df.head())

    # Save cleaned data to session state for model training page
    st.session_state["cleaned_data"] = df

    # Download Cleaned Data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Processed CSV", data=csv, file_name='processed_data.csv', mime='text/csv')

else:
    st.info("📌 Please upload a CSV file to get started.")
