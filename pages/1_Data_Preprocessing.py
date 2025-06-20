import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import helpers.eda as eda
from helpers import preprocess

# --- Page Setup ---
st.set_page_config(page_title="No-Code ML Builder", layout="wide")
st.title("No-Code Machine Learning Builder")
st.markdown("Upload your dataset and begin your no-code ML journey. This tool guides you step-by-step from raw data to model training without writing code.")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["raw_data"] = df.copy()
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        st.stop()

    st.info("✅ File uploaded successfully. Start by exploring or preprocessing the data.")

    # --- Optional: Drop Unwanted Columns ---
    st.subheader("Drop Unwanted Columns")
    cols_to_drop = st.multiselect("Select columns to remove from the dataset (e.g., IDs, timestamps):", df.columns.tolist())
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        st.success(f"Removed columns: {', '.join(cols_to_drop)}")

    # --- Dataset Overview ---
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Column Types")
    st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    # --- Smart Suggestions ---
    st.markdown("### Smart Suggestions")
    suggestions = []

    if df.isnull().sum().sum() > 0:
        st.warning("Missing values found. Consider filling or dropping them before model training.")
        suggestions.append("➡️ Handle missing values under the preprocessing section.")

    if df.select_dtypes(include='object').shape[1] > 0:
        st.warning("Categorical columns detected. Encoding is required for ML models.")
        suggestions.append("➡️ Use One-Hot Encoding for text columns before training.")

    if df.select_dtypes(include='number').shape[1] > 0:
        std_range = df.describe().T[["min", "max"]]
        if (std_range["max"] - std_range["min"]).max() > 1000:
            st.info("Large numeric range detected. Scaling is recommended.")
            suggestions.append("➡️ Use Standard or MinMax Scaler.")

    if df.duplicated().sum() > 0:
        st.warning(f"{df.duplicated().sum()} duplicate rows found.")
        suggestions.append("➡️ Consider removing duplicates.")

    if df.shape[0] < 10:
        st.error("Dataset too small for ML. Please upload more rows.")
    elif len(suggestions) == 0:
        st.success("Dataset looks clean! Proceed to preprocessing or training.")

    for s in suggestions:
        st.markdown(s)

    # --- Exploratory Data Analysis ---
    st.markdown("---")
    st.header("Exploratory Data Analysis")

    with st.expander("Descriptive Statistics"):
        eda.show_basic_info(df)

    with st.expander("Missing Values"):
        eda.show_missing_values(df)

    with st.expander("Correlation Heatmap"):
        eda.show_correlation_heatmap(df)

    with st.expander("Distribution Plots"):
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns found for plotting.")
        else:
            for col in numeric_cols:
                try:
                    if df[col].nunique() > 10000:
                        st.warning(f"Skipping '{col}' due to high cardinality.")
                        continue
                    fig, ax = plt.subplots()
                    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax)
                    ax.set_title(f"Distribution of {col}")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not plot '{col}': {e}")

    with st.expander("Count Plots"):
        eda.show_categorical_counts(df)

    # --- Data Preprocessing ---
    st.markdown("---")
    st.header("Data Preprocessing")

    with st.expander("Preprocessing Options"):
        # Handle Missing Values
        st.subheader("Handle Missing Values")
        st.markdown("Missing values can reduce model accuracy. Choose how to handle them:")
        missing_option = st.selectbox("Strategy", ['drop', 'mean', 'fill'])
        fill_val = None
        if missing_option == 'fill':
            fill_val = st.text_input("Fill with value:", "0")
            df = preprocess.handle_missing_values(df, method=missing_option, fill_value=fill_val)
        else:
            df = preprocess.handle_missing_values(df, method=missing_option)

        # Encode Categorical Columns
        st.subheader("Encode Categorical Columns")
        st.markdown("Convert text/categorical columns into numbers using One-Hot Encoding.")
        max_classes = st.slider("Max Unique Values for One-Hot Encoding", 2, 50, 10)
        df = preprocess.encode_categoricals(df, max_onehot_classes=max_classes)

        # Scale Numerical Features
        st.subheader("Scale Numerical Features")
        st.markdown("Scale features for better model performance. Recommended for Logistic Regression, KNN, SVM, etc.")
        scale_option = st.selectbox("Scaling Method", ['none', 'standard', 'minmax'])
        if scale_option != 'none':
            df = preprocess.scale_numerical(df, method=scale_option)

    # --- Data Consistency & Arrow Compatibility Fix ---
    st.subheader("Transformed Data Preview")

    # Convert all non-numeric and mixed-type columns to string (for Streamlit safety)
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].apply(type).nunique() > 1:
            df[col] = df[col].astype(str)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            df[col] = df[col].astype(str)

    for col in df.columns:
        types = df[col].map(type).value_counts()
        if len(types) > 1:
            st.warning(f"Column '{col}' contains mixed types: {types.to_dict()}")

    st.dataframe(df.sample(min(len(df), 10)))

    # Save cleaned data
    st.session_state["cleaned_data"] = df

    # --- Final Options ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Processed CSV", data=csv, file_name='processed_data.csv', mime='text/csv')

    st.markdown("---")
    st.success("Data is preprocessed and ready for model training.")
    st.markdown("➡️ You can now go to the **Model Trainer** tab to build and evaluate your ML model.")

else:
    st.info("Please upload a CSV file to get started.")
