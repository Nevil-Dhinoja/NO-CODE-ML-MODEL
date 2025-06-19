# model_trainer.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import shap
import xgboost as xgb
import lightgbm as lgb

st.set_page_config(page_title="Train & Predict ML Models", layout="wide")
st.title("🚀 Advanced ML Model Trainer & Prediction Playground")

if "cleaned_data" not in st.session_state:
    st.warning("⚠️ Please preprocess data first on the previous page.")
    st.stop()

df = st.session_state["cleaned_data"]

st.write("### Dataset Preview")
st.dataframe(df.head())

target_col = st.selectbox("Select the target column (label):", df.columns)

X = df.drop(columns=[target_col])
y = df[target_col]

is_classification = y.dtype == "object" or (y.nunique() < 20 and y.dtype in [int, float])

if is_classification:
    st.info("Detected Classification Problem")
else:
    st.info("Detected Regression Problem")

# Encode target if classification and categorical
if is_classification and y.dtype == "object":
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
else:
    y_enc = y
    le = None

# Split dataset
test_size = st.slider("Test set size (percentage):", 10, 50, 20)
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size / 100, random_state=random_state)

st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Standardize numeric features if needed
scale_data = st.checkbox("Scale numeric features (recommended for SVM, Logistic Regression)", value=True)
if scale_data:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
else:
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

# Model selection
st.write("### Select Model")

models = []
if is_classification:
    models = [
        "Logistic Regression",
        "Random Forest Classifier",
        "Decision Tree Classifier",        # Added
        "K-Nearest Neighbors Classifier",  # Added
        "Gradient Descent (SGD Classifier)",  # New
        "Support Vector Machine",
        "XGBoost Classifier",
        "LightGBM Classifier"
    ]
else:
    models = [
        "Random Forest Regressor",
        "Decision Tree Regressor",         # Added
        "K-Nearest Neighbors Regressor",   # Added
        "Gradient Descent (SGD Regressor)",   # New
        "XGBoost Regressor",
        "LightGBM Regressor"
    ]

model_name = st.selectbox("Choose Model:", models)

# Hyperparameter tuning method selection
tuning_method = st.selectbox("Hyperparameter Tuning Method:", ["None", "Grid Search", "Randomized Search"])

# Define models and param grids for search
def get_model_and_params(name):
    if name == "Logistic Regression":
        model = LogisticRegression(max_iter=500, random_state=random_state)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    elif name == "Random Forest Classifier":
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    elif name == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=random_state)
        param_grid = {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif name == "K-Nearest Neighbors Classifier":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    elif name == "Support Vector Machine":
        model = SVC(probability=True, random_state=random_state)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    elif name == "XGBoost Classifier":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif name == "LightGBM Classifier":
        model = lgb.LGBMClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif name == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    elif name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=random_state)
        param_grid = {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif name == "K-Nearest Neighbors Regressor":
        model = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    elif name == "XGBoost Regressor":
        model = xgb.XGBRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif name == "LightGBM Regressor":
        model = lgb.LGBMRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif name == "Gradient Descent (SGD Classifier)":
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(random_state=random_state)
        param_grid = {
            'loss': ['hinge', 'log_loss', 'modified_huber'],
            'alpha': [0.0001, 0.001, 0.01],
            'penalty': ['l2', 'l1', 'elasticnet']
        }
    elif name == "Gradient Descent (SGD Regressor)":
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor(random_state=random_state)
        param_grid = {
            'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
            'alpha': [0.0001, 0.001, 0.01],
            'penalty': ['l2', 'l1', 'elasticnet']
        }

    else:
        model = None
        param_grid = {}

    return model, param_grid

model, default_param_grid = get_model_and_params(model_name)

# Manual hyperparameter input UI if tuning_method == "None"
manual_params = {}
if tuning_method == "None":
    st.write("### Set Hyperparameters Manually")

    if model_name == "Logistic Regression":
        C = st.number_input("C (Inverse Regularization strength)", min_value=0.0001, max_value=100.0, value=1.0, step=0.1, format="%.4f")
        solver = st.selectbox("Solver", options=['liblinear', 'lbfgs'])
        manual_params = {'C': C, 'solver': solver, 'max_iter': 500, 'random_state': random_state}

    elif model_name in ["Random Forest Classifier", "Random Forest Regressor"]:
        n_estimators = st.slider("Number of Trees (n_estimators)", min_value=10, max_value=500, value=100, step=10)
        max_depth_options = [None, 5, 10, 20, 50]
        max_depth = st.selectbox("Max Depth", options=max_depth_options, index=1)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
        manual_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state
        }

    elif model_name in ["Decision Tree Classifier", "Decision Tree Regressor"]:
        max_depth_options = [None, 5, 10, 20, 50]
        max_depth = st.selectbox("Max Depth", options=max_depth_options, index=1)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
        manual_params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state
        }

    elif model_name in ["K-Nearest Neighbors Classifier", "K-Nearest Neighbors Regressor"]:
        n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5)
        weights = st.selectbox("Weights", options=['uniform', 'distance'])
        p = st.selectbox("Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)", options=[1, 2], index=1)
        manual_params = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'p': p
        }

    elif model_name == "Support Vector Machine":
        C = st.number_input("C (Regularization parameter)", min_value=0.01, max_value=100.0, value=1.0, step=0.1, format="%.4f")  
        kernel = st.selectbox("Kernel", options=['linear', 'rbf', 'poly'])
        gamma = st.selectbox("Gamma", options=['scale', 'auto'])
        manual_params = {'C': C, 'kernel': kernel, 'gamma': gamma, 'probability': True, 'random_state': random_state}

    elif model_name in ["XGBoost Classifier", "XGBoost Regressor"]:
        n_estimators = st.slider("Number of Trees (n_estimators)", min_value=10, max_value=500, value=100, step=10)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=6, step=1)
        learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
        manual_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'use_label_encoder': False
        }
        if is_classification:
            manual_params['eval_metric'] = 'logloss'

    elif model_name in ["LightGBM Classifier", "LightGBM Regressor"]:
        n_estimators = st.slider("Number of Trees (n_estimators)", min_value=10, max_value=500, value=100, step=10)
        max_depth = st.slider("Max Depth (-1 means no limit)", min_value=-1, max_value=20, value=6, step=1)
        learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
        manual_params = {
            'n_estimators': n_estimators,
            'max_depth': None if max_depth == -1 else max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
    elif model_name == "Gradient Descent (SGD Classifier)":
        from sklearn.linear_model import SGDClassifier
        loss = st.selectbox("Loss", options=['hinge', 'log_loss', 'modified_huber'])
        alpha = st.number_input("Alpha (regularization strength)", min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001, format="%.4f")
        penalty = st.selectbox("Penalty", options=['l2', 'l1', 'elasticnet'])
        manual_params = {
            'loss': loss,
            'alpha': alpha,
            'penalty': penalty,
            'random_state': random_state
        }

    elif model_name == "Gradient Descent (SGD Regressor)":
        from sklearn.linear_model import SGDRegressor
        loss = st.selectbox("Loss", options=['squared_error', 'huber', 'epsilon_insensitive'])
        alpha = st.number_input("Alpha (regularization strength)", min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001, format="%.4f")
        penalty = st.selectbox("Penalty", options=['l2', 'l1', 'elasticnet'])
        manual_params = {
            'loss': loss,
            'alpha': alpha,
            'penalty': penalty,
            'random_state': random_state
        }

    # Instantiate model with manual params
    try:
        model = model.__class__(**manual_params)
    except Exception as e:
        st.error(f"Error creating model with parameters: {e}")

# Setup tuning search if selected
search_cv = None
if tuning_method == "Grid Search":
    search_cv = GridSearchCV(model, default_param_grid, cv=3, n_jobs=-1, verbose=1)
elif tuning_method == "Randomized Search":
    from scipy.stats import uniform, randint
    search_cv = RandomizedSearchCV(model, default_param_grid, cv=3, n_iter=10, n_jobs=-1, verbose=1, random_state=random_state)

# Train model button
if st.button("Train Model"):

    with st.spinner("Training the model..."):

        if search_cv is not None:
            search_cv.fit(X_train, y_train)
            best_model = search_cv.best_estimator_
            st.success(f"Best params: {search_cv.best_params_}")
        else:
            model.fit(X_train, y_train)
            best_model = model

    st.success("✅ Model trained successfully!")

    # Prediction and metrics
    y_pred = best_model.predict(X_test)

    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.4f}")

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Decode predictions if LabelEncoder used
        if le is not None:
            y_pred_labels = le.inverse_transform(y_pred)
            y_test_labels = le.inverse_transform(y_test)
        else:
            y_pred_labels = y_pred
            y_test_labels = y_test

    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**Mean Squared Error:** {mse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

    # Feature Importance / Interpretability
    st.write("### Model Interpretability")

    try:
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)

            fig2, ax2 = plt.subplots()
            sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax2)
            ax2.set_title("Feature Importances")
            st.pyplot(fig2)

        elif 'xgb' in model_name.lower() or 'lightgbm' in model_name.lower():
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test if not scale_data else scaler.inverse_transform(X_test))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("SHAP Summary Plot:")
            shap.summary_plot(shap_values, X_test if not scale_data else pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns), plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')

        else:
            st.info("Feature importance not available for this model.")
    except Exception as e:
        st.warning(f"Feature importance plot failed: {e}")

    # Save model and scaler
    model_data = {'model': best_model, 'label_encoder': le}
    if scale_data:
        model_data['scaler'] = scaler

    model_bytes = io.BytesIO()
    pickle.dump(model_data, model_bytes)
    model_bytes.seek(0)

    st.download_button("⬇️ Download Trained Model (pickle)", data=model_bytes, file_name="trained_model.pkl", mime="application/octet-stream")

    # Save model to session state for prediction playground
    st.session_state['trained_model'] = best_model
    st.session_state['label_encoder'] = le
    st.session_state['scaler'] = scaler if scale_data else None
    st.session_state['X_columns'] = X.columns.tolist()

else:
    st.info("Press the Train Model button to start training.")
import streamlit as st
import pandas as pd
import numpy as np

st.markdown("---")
st.header("🧪 Prediction Playground")

if 'trained_model' not in st.session_state:
    st.warning("Train a model first to use the prediction playground.")
    st.stop()

model = st.session_state['trained_model']
scaler = st.session_state.get('scaler', None)
encoder = st.session_state.get('encoder', None)
le = st.session_state.get('label_encoder', None)
X_columns = st.session_state['X_columns']
is_classification = st.session_state.get('is_classification', False)
df = st.session_state['cleaned_data']

st.write("Enter values for each feature below:")

input_data = {}
for col in X_columns:
    dtype = df[col].dtype
    if np.issubdtype(dtype, np.number):
        val = st.number_input(f"{col} (numeric)", value=float(df[col].mean()))
    else:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) <= 10:
            val = st.selectbox(f"{col} (categorical)", options=unique_vals)
        else:
            val = st.text_input(f"{col} (categorical/text)", value=str(df[col].mode()[0]))
    input_data[col] = val

input_df = pd.DataFrame([input_data])

# Align columns
input_df = input_df[X_columns]

# Handle categorical encoding using OneHotEncoder
if encoder is not None:
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    try:
        input_encoded = encoder.transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(input_encoded.toarray(), columns=encoder.get_feature_names_out(cat_cols))
        input_df = input_df.drop(columns=cat_cols).reset_index(drop=True)
        input_df = pd.concat([input_df, encoded_df], axis=1)
    except Exception as e:
        st.error(f"Encoding failed: {e}")
        st.stop()

# Apply scaler
if scaler is not None:
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Input scaling failed: {e}")
        input_scaled = input_df.values
else:
    input_scaled = input_df.values

# Predict button
if st.button("Predict"):
    try:
        pred = model.predict(input_scaled)
        if is_classification and le is not None:
            pred_label = le.inverse_transform(pred)[0]
            st.success(f"Predicted Class: {pred_label}")
        else:
            st.success(f"Predicted Value: {pred[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# st.markdown("---")
# st.header("🧪 Prediction Playground")

# if 'trained_model' not in st.session_state:
#     st.warning("Train a model first to use the prediction playground.")
#     st.stop()

# model = st.session_state['trained_model']
# le = st.session_state['label_encoder']
# scaler = st.session_state['scaler']
# X_columns = st.session_state['X_columns']

# st.write("Enter values for each feature below:")

# input_data = {}
# for col in X_columns:
#     dtype = df[col].dtype
#     if np.issubdtype(dtype, np.number):
#         val = st.number_input(f"{col} (numeric)", value=float(df[col].mean()))
#     else:
#         unique_vals = df[col].unique().tolist()
#         if len(unique_vals) <= 10:
#             val = st.selectbox(f"{col} (categorical)", options=unique_vals)
#         else:
#             val = st.text_input(f"{col} (categorical/text)", value=str(df[col].mode()[0]))
#     input_data[col] = val

# input_df = pd.DataFrame([input_data])

# # Encode categorical input same way as training data
# for col in input_df.columns:
#     if input_df[col].dtype == 'object' and col in df.select_dtypes(include='object').columns:
#         input_df[col] = input_df[col].astype(str)

# # Scale input if scaler used
# if scaler is not None:
#     try:
#         input_scaled = scaler.transform(input_df)
#     except Exception as e:       
#         st.error(f"Input scaling failed: {e}")
#         input_scaled = input_df.values
# else:
#     input_scaled = input_df.values

# if st.button("Predict"):
#     try:
#         pred = model.predict(input_scaled)

#         if is_classification and le is not None:
#             pred_label = le.inverse_transform(pred)[0]
#             st.success(f"Predicted Class: {pred_label}")
#         else:
#             st.success(f"Predicted Value: {pred[0]}")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
       
