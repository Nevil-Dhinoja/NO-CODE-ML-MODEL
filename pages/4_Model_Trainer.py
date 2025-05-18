import streamlit as st
import pandas as pd
from helpers.model import train_automl, train_manual_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import os

st.title("🤖 Model Builder & Trainer")

if 'cleaned_data' not in st.session_state:
    st.warning("Please preprocess your data first on the main page.")
    st.stop()

df = st.session_state.cleaned_data

target = st.selectbox("🎯 Select Target Column", df.columns)
features = st.multiselect("🧩 Select Feature Columns", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

X = df[features]
y = df[target]

problem_type = "classification" if y.nunique() <= 20 else "regression"
st.info(f"🔍 Detected Task: **{problem_type.title()}**")

tab1, tab2 = st.tabs(["🚀 AutoML (LazyPredict)", "🔧 Manual Model Builder"])

with tab1:  
    st.subheader("🚀 AutoML Training")
    test_size = st.slider("📏 Test Size", 0.1, 0.5, 0.2, 0.05)
    
    if st.button("Run AutoML"):
        with st.spinner("Training..."):
            results = train_automl(X, y, problem_type, test_size)
        st.success("Done!")
        
        if problem_type == "classification":
            metric_col = next((col for col in ["Accuracy", "F1 Score", "ROC AUC", "Balanced Accuracy"] if col in results.columns), None)
        else:
            metric_col = next((col for col in ["R2", "R2 Score", "RMSE", "MAE"] if col in results.columns), None)

        if metric_col:
            st.dataframe(results.sort_values(metric_col, ascending=False))
        else:
            st.warning("⚠️ No suitable metric column found to sort the results.")
            st.dataframe(results)



with tab2:
    st.subheader("🔧 Train a Model Manually")

    model_options = {
        "classification": {
            "Logistic Regression": LogisticRegression,
            "Random Forest": RandomForestClassifier,
            "KNN": KNeighborsClassifier,
            "SVM": SVC
        },
        "regression": {
            "Linear Regression": LinearRegression,
            "Random Forest": RandomForestRegressor,
            "KNN": KNeighborsRegressor,
            "SVR": SVR
        }
    }

    model_name = st.selectbox("📦 Select Model", list(model_options[problem_type].keys()))
    ModelClass = model_options[problem_type][model_name]

    # Hyperparameter tuning UI
    if "Random Forest" in model_name:
        n_estimators = st.slider("🌲 n_estimators", 10, 200, 100)
        model = ModelClass(n_estimators=n_estimators)
    elif "KNN" in model_name:
        k = st.slider("👥 Neighbors", 1, 20, 5)
        model = ModelClass(n_neighbors=k)
    else:
        model = ModelClass()

    test_size_manual = st.slider("📏 Test Size", 0.1, 0.5, 0.2, 0.05, key="manual")

    if st.button("Train Manually"):
        with st.spinner("Training..."):
            result = train_manual_model(model, X, y, problem_type, test_size_manual)

        st.success("Model Trained!")

        if problem_type == "classification":
            st.metric("🎯 Accuracy", f"{result['accuracy']:.2f}")
            st.metric("📊 F1 Score", f"{result['f1_score']:.2f}")

            st.write("🧮 Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(result["confusion_matrix"]).plot(ax=ax)
            st.pyplot(fig)

        else:
            st.metric("📈 R² Score", f"{result['r2_score']:.2f}")
            st.metric("📉 RMSE", f"{result['rmse']:.2f}")

        # Save model to session
        st.session_state['model'] = result['model']

        # Save model to disk
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(result['model'], model_path)
        st.info(f"Model saved as {model_path}")

        # Download button
        with open(model_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Trained Model",
                data=f,
                file_name=model_path,
                mime="application/octet-stream"
            )
