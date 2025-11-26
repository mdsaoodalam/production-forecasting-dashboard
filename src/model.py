import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def train_cart_decision_tree(df, target_column, model_type="classification",
                             test_size=0.2, max_depth=None, min_samples_split=2,
                             criterion="gini"):

    # ---------------------------
    # Split X and y
    # ---------------------------
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # ---------------------------
    # Convert non-numeric features
    # ---------------------------
    for col in X.select_dtypes(include='object').columns:
        # Try parsing month-year strings
        X[col] = pd.to_datetime(X[col], errors="coerce").map(lambda x: x.timestamp() if pd.notna(x) else x)
        # If parsing failed for any row, fall back to label encoding
        if X[col].isna().any():
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # ---------------------------
    # Encode target column
    # ---------------------------
    if model_type == "classification":
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
    else:  # regression
        if y.dtype == "object":
            # Try converting month-year strings to timestamps
            y = pd.to_datetime(y, errors="coerce").map(lambda x: x.timestamp() if pd.notna(x) else x)
            if y.isna().any():
                raise ValueError(
                    f"Target column '{target_column}' contains non-numeric strings. "
                    "Please select a numeric column for regression or convert it to dates."
                )

    # ---------------------------
    # Train / Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ---------------------------
    # Initialize model
    # ---------------------------
    if model_type == "classification":
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    else:
        model = DecisionTreeRegressor(
            criterion="squared_error",  # CART default for regression
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

    # ---------------------------
    # Train model
    # ---------------------------
    model.fit(X_train, y_train)

    # ---------------------------
    # Evaluate performance
    # ---------------------------
    y_pred = model.predict(X_test)
    if model_type == "classification":
        score = accuracy_score(y_test, y_pred)
        metrics = f"Accuracy: {score:.4f}"
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = f"MSE: {mse:.4f} | R² Score: {r2:.4f}"

    return model, metrics


def save_model(model, filename="decision_tree_model.pkl"):
    path = os.path.join("models", filename)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    return path
