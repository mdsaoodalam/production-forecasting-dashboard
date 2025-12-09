import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.model import train_cart_decision_tree, save_model


def render_train_tab(df_to_use: pd.DataFrame) -> None:
    """
    Render the training tab UI for training a CART decision tree model.

    Parameters
    ----------
    df_to_use : pd.DataFrame
        The dataset provided in the previous step.
    """
    st.info(
        "Here, you can train machine learning models on the selected dataset, "
        "configure hyperparameters, and monitor the training process to achieve "
        "the best performance."
    )

    if not df_to_use.empty:
        st.subheader("🌲 Train CART Decision Tree Model")

        # Model type: classification or regression
        model_type = st.radio("Model Type:", ["classification", "regression"])

        # Select target column
        if model_type == "regression":
            numeric_columns = df_to_use.select_dtypes(include="number").columns.tolist()
            target_column = st.selectbox("Select target column:", numeric_columns)
        else:
            target_column = st.selectbox("Select target column:", df_to_use.columns)

        # Hyperparameters
        st.markdown("### ⚙ Hyperparameters")

        col1, col2, col3 = st.columns(3)
        with col1:
            max_depth = st.number_input(
                "Max Depth (None = auto)", min_value=1, value=5
            )
        with col2:
            min_samples_split = st.number_input(
                "Min Samples Split", min_value=2, value=2
            )
        with col3:
            test_size = st.slider("Test Size %", 5, 40, 20) / 100

        criterion = "gini" if model_type == "classification" else "squared_error"

        # Train button
        if st.button("🚀 Train Decision Tree Model"):

            # Prepare features and target
            X = df_to_use.drop(columns=[target_column]).copy()
            y = df_to_use[target_column].copy()

            # Convert non-numeric X columns
            for col in X.select_dtypes(include="object").columns:
                try:
                    X[col] = pd.to_datetime(
                        X[col], format="%m-%Y"
                    ).map(lambda x: x.timestamp())
                except Exception:
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col])

            # Encode target if classification and non-numeric
            if model_type == "classification" and y.dtype == "object":
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)

            # Convert month-year regression target to timestamps
            if model_type == "regression" and y.dtype == "object":
                try:
                    y = pd.to_datetime(
                        y, format="%m-%Y"
                    ).map(lambda x: x.timestamp())
                except Exception:
                    st.error(
                        "Selected target column is non-numeric. "
                        "Please choose a numeric column for regression."
                    )
                    st.stop()

            # Train the model
            model, metrics = train_cart_decision_tree(
                df=df_to_use,
                target_column=target_column,
                model_type=model_type,
                test_size=test_size,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
            )

            st.success("🎉 Model trained successfully!")
            st.info(f"📈 Performance → {metrics}")

            # Save model for prediction tab
            st.session_state["trained_model"] = model
            st.session_state["trained_features"] = X.columns.tolist()
            st.session_state["trained_target"] = target_column

            # Save button
            if st.button("💾 Save Model"):
                path = save_model(model)
                st.success(f"Model saved to: `{path}`")
