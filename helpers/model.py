# helpers/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics import mean_absolute_error

from lazypredict.Supervised import LazyClassifier, LazyRegressor

# AutoML Training
def train_automl(X, y, problem_type='classification', test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    if problem_type == 'classification':
        model = LazyClassifier()
    else:
        model = LazyRegressor()

    results, _ = model.fit(X_train, X_test, y_train, y_test)
    return results

    
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_squared_error
)
import numpy as np

def train_manual_model(model, X, y, problem_type, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        return {
            "model": model,
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm
        }

    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        residuals = y_test - y_pred

        return {
            "model": model,
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae,
            "y_test": y_test,
            "y_pred": y_pred,
            "residuals": residuals
        }
