# No-Code Machine Learning Builder
# This is a simple Streamlit app that allows users to upload a CSV file,
# perform basic data preprocessing, and prepare the data for machine learning model training.
# app.py 
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
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["raw_data"] = df.copy()
    except Exception as e:
        st.error(f"❌ Failed to read CSV file: {e}")
        st.stop()

    # --- Optional: Drop Unwanted Columns ---
    st.subheader("🧯 Drop Unwanted Columns")
    cols_to_drop = st.multiselect("Select columns to remove from the dataset:", df.columns.tolist())
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        st.success(f"Removed columns: {', '.join(cols_to_drop)}")

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

    # --- Smart Suggestions ---
    st.markdown("### 🧠 Smart Suggestions")
    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Your data contains missing values. Consider handling them before training.")

    if df.select_dtypes(include='object').shape[1] > 0:
        st.warning("⚠️ Categorical columns detected. Apply encoding to use with most ML models.")

    if df.select_dtypes(include='number').shape[1] > 0:
        std_range = df.describe().T[["min", "max"]]
        if (std_range["max"] - std_range["min"]).max() > 1000:
            st.info("ℹ️ Large numeric range detected. Scaling is recommended for SVM, KNN, Logistic Regression, etc.")

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

    with st.expander("⚙️ Preprocessing Options"):
        # Handle Missing Values
        st.subheader("🧩 Handle Missing Values")
        missing_option = st.selectbox(
            "Choose missing value strategy", ['drop', 'mean', 'fill'],
            help="Use 'drop' if missing values are few. Use 'mean' or 'fill' to retain more data."
        )
        fill_val = None
        if missing_option == 'fill':
            fill_val = st.text_input("Fill missing values with:", "0")
            df = preprocess.handle_missing_values(df, method=missing_option, fill_value=fill_val)
        else:
            df = preprocess.handle_missing_values(df, method=missing_option)

        # Encode Categorical Columns
        st.subheader("🔤 Encode Categorical Columns")
        max_classes = st.slider(
            "Max Unique Values for One-Hot Encoding", 2, 50, 10,
            help="If a column has more categories than this, it will be skipped from encoding."
        )
        df = preprocess.encode_categoricals(df, max_onehot_classes=max_classes)

        # Scale Numerical Features
        st.subheader("📏 Scale Numerical Features")
        scale_option = st.selectbox(
            "Choose scaling method", ['none', 'standard', 'minmax'],
            help="Use scaling for algorithms like SVM, KNN, or Logistic Regression. Tree-based models don't need it."
        )
        if scale_option != 'none':
            df = preprocess.scale_numerical(df, method=scale_option)

    # --- Ensure Consistent Dtypes ---
    st.subheader("🧾 Transformed Data Preview")

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (bool, int))).any():
            try:
                df[col] = df[col].astype(int)
            except:
                df[col] = df[col].astype(str)
        elif df[col].apply(type).nunique() > 1:
            df[col] = df[col].astype(str)

    for col in df.columns:
        types = df[col].map(type).value_counts()
        if len(types) > 1:
            st.warning(f"⚠️ Column '{col}' has multiple types: {types.to_dict()}")

    # Show preview sample
    st.dataframe(df.sample(10))

    # Save cleaned data to session state for training
    st.session_state["cleaned_data"] = df

    # Download Processed Data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Processed CSV", data=csv, file_name='processed_data.csv', mime='text/csv')

    # Final CTA: Go to Model Training
    st.markdown("---")
    st.success("✅ Data preprocessed and ready for training!")
    st.button("🚀 Go to Model Training"):
    # st.switch_page("pages/3_model_trainer.py")
else:
    st.info("📌 Please upload a CSV file to get started.")
