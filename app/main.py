import os
import sys

import pandas as pd
import streamlit as st

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.plots import (
    barplot,
    boxplot,
    catplot,
    distplot,
    ecdfplot,
    histplot,
    kdeplot,
    lineplot,
    pointplot,
    rugplot,
    scatterplot,
    stripplot,
    swarmplot,
    violinplot,
)

from strings.strings import (
    DESC_CUMULATIVE_OIL_PRODUCTION_2020,
    DESC_MCF_GAS_PRODUCTION_BY_COUNTY,
    DESC_MONTHLY_OIL_PRODUCTION_BY_COUNTY,
    DESC_NATURAL_GAS_PRODUCTION,
    NAME_CUMULATIVE_OIL_PRODUCTION_2020,
    NAME_MCF_GAS_PRODUCTION_BY_COUNTY,
    NAME_MONTHLY_OIL_PRODUCTION_BY_COUNTY,
    NAME_NATURAL_GAS_PRODUCTION,
    PLATFORM_DESCRIPTION,
    UPLOAD_DATASET_INFO,
    UPLOAD_INSTRUCTION,
    WELCOME_MESSAGE,
    WORKFLOW_SUPPORT_MESSAGE,
)


save_dir = os.path.join("data") 

# Ensure the folder exists
os.makedirs(save_dir, exist_ok=True)

dataset_1 = False
dataset_2 = False
dataset_3 = False
dataset_4 = False

home, upload_dataset, data_eng_tab, train_ml_model, prediction_tab = st.tabs([
    "🏠 Home", 
    "📁 Select/Upload Dataset", 
    "🛠️ Data Engineering", 
    "🤖 Train ML Model",
    "🔮 Prediction"
])


with home:
    st.title("Welcome to Well Production Forecasting Dashboard 🛢️📈")
    st.write(WELCOME_MESSAGE)
    st.write(PLATFORM_DESCRIPTION)
    st.write(UPLOAD_INSTRUCTION)
    st.write(WORKFLOW_SUPPORT_MESSAGE)


with upload_dataset:
    st.info(UPLOAD_DATASET_INFO)

    choice = st.radio(
        "Choose one option:",
        ["I want to upload my dataset.", "I want to select from the real-world datasets."]
    )

    if choice == "I want to upload my dataset.":
        st.write("====================================================")
        st.info("You can upload only `.xlsx` files.")
        number_of_user_dataset = 1
        uploaded_file = st.file_uploader("Upload your dataset", type=["xlsx"])
        if uploaded_file:
            df_uploaded_file = pd.read_excel(uploaded_file)
            st.session_state['uploaded_dataset'] = df_uploaded_file
            st.session_state['uploaded_filename'] = uploaded_file.name
            st.dataframe(df_uploaded_file)


    elif choice == "I want to select from the real-world datasets.":
        st.write("====================================================")
        DATASET_NAMES = [
            NAME_NATURAL_GAS_PRODUCTION,
            NAME_CUMULATIVE_OIL_PRODUCTION_2020,
            NAME_MONTHLY_OIL_PRODUCTION_BY_COUNTY,
            NAME_MCF_GAS_PRODUCTION_BY_COUNTY
        ]

        selected_dataset = st.radio("Select a dataset:", DATASET_NAMES)

        dataset_descriptions = {
            NAME_NATURAL_GAS_PRODUCTION: DESC_NATURAL_GAS_PRODUCTION,
            NAME_CUMULATIVE_OIL_PRODUCTION_2020: DESC_CUMULATIVE_OIL_PRODUCTION_2020,
            NAME_MONTHLY_OIL_PRODUCTION_BY_COUNTY: DESC_MONTHLY_OIL_PRODUCTION_BY_COUNTY,
            NAME_MCF_GAS_PRODUCTION_BY_COUNTY: DESC_MCF_GAS_PRODUCTION_BY_COUNTY,
        }

        with st.expander("Dataset Description"):
            st.write(dataset_descriptions[selected_dataset])


def load_default_dataset(name):
    if name == "North Dakota Natural Gas Production":
        return pd.read_excel("data/ND_gas_1990_to_present.xlsx")
    elif name == "North Dakota Cumulative Oil Production by Formation Through 2020":
        return pd.read_excel("data/ND_cumulative_formation_2020.xlsx")
    elif name == "North Dakota Historical Monthly Oil Production by County":
        return pd.read_excel("data/ND_historical_barrels_of_oil_produced_by_county.xlsx")
    elif name == "North Dakota Historical MCF Gas Produced by County":
        return pd.read_excel("data/ND_historical_MCF_gas_produced_by_county.xlsx")
    else:
        return pd.DataFrame()


with data_eng_tab:
    st.info("Here, you can visualize and process your selected dataset before training your model.")

    # Dataset selection
    if choice == "I want to upload my dataset." and 'uploaded_dataset' in st.session_state:
        st.success(f"✅ You selected your uploaded dataset: {st.session_state['uploaded_filename']}")
        df_to_use = st.session_state['uploaded_dataset']

    elif choice == "I want to select from the real-world datasets.":
        st.success(f"🌍 You selected: {selected_dataset}")
        df_to_use = load_default_dataset(selected_dataset)

    else:
        st.warning("⚠️ No dataset selected yet.")
        df_to_use = pd.DataFrame()

    # If dataset is available
    if not df_to_use.empty:
        st.subheader("📊 Dataset Overview")
        st.dataframe(df_to_use.describe())

        # Master toggle for visualization
        if st.toggle("📈 Enable Data Visualization"):
            st.markdown("### 🎨 Choose Plot Group")

            # Group selector
            plot_group = st.selectbox(
                "Select a group of plots:",
                ["Distribution Plots", "Categorical Plots", "Relational Plots"]
            )

            df_columns = df_to_use.columns.tolist()

            # --- Distribution Plots ---
            if plot_group == "Distribution Plots":
                with st.expander("Distribution Plot"):
                    x = st.radio("Select column:", df_columns, index=0, key="dist_x")
                    st.info(f"Plotting distribution of **{x}**")
                    distplot(df_to_use, x=x)

                with st.expander("Histogram"):
                    x = st.radio("Select column:", df_columns, index=0, key="hist_x")
                    bins = st.slider("Number of bins", 5, 100, 10, key="hist_bins")
                    st.info(f"Plotting histogram of **{x}** with {bins} bins")
                    histplot(df_to_use, x=x, bins=bins)

                with st.expander("KDE Plot"):
                    x = st.radio("Select column:", df_columns, index=0, key="kde_x")
                    st.info(f"Plotting KDE of **{x}**")
                    kdeplot(df_to_use, x=x)

                with st.expander("ECDF Plot"):
                    x = st.radio("Select column:", df_columns, index=0, key="ecdf_x")
                    st.info(f"Plotting ECDF of **{x}**")
                    ecdfplot(df_to_use, x=x)

                with st.expander("Rug Plot"):
                    x = st.radio("Select column:", df_columns, index=0, key="rug_x")
                    st.info(f"Plotting rug plot of **{x}**")
                    rugplot(df_to_use, x=x)

            # --- Categorical Plots ---
            elif plot_group == "Categorical Plots":
                with st.expander("Cat Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="cat_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="cat_y")
                    st.info(f"Plotting categorical **{y}** vs **{x}**")
                    catplot(df_to_use, x=x, y=y, kind="box")

                with st.expander("Strip Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="strip_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="strip_y")
                    st.info(f"Plotting strip **{y}** vs **{x}**")
                    stripplot(df_to_use, x=x, y=y)

                with st.expander("Swarm Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="swarm_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="swarm_y")
                    st.info(f"Plotting swarm **{y}** vs **{x}**")
                    swarmplot(df_to_use, x=x, y=y)

                with st.expander("Box Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="box_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="box_y")
                    st.info(f"Plotting box **{y}** vs **{x}**")
                    boxplot(df_to_use, x=x, y=y)

                with st.expander("Violin Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="violin_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="violin_y")
                    st.info(f"Plotting violin **{y}** vs **{x}**")
                    violinplot(df_to_use, x=x, y=y)

                with st.expander("Point Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="point_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="point_y")
                    st.info(f"Plotting point **{y}** vs **{x}**")
                    pointplot(df_to_use, x=x, y=y)

                with st.expander("Bar Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="bar_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="bar_y")
                    st.info(f"Plotting bar **{y}** vs **{x}**")
                    barplot(df_to_use, x=x, y=y)

            # --- Relational Plots ---
            elif plot_group == "Relational Plots":
                with st.expander("Scatter Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="scatter_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="scatter_y")
                    st.info(f"Plotting **{y}** vs **{x}**")
                    scatterplot(df_to_use, x=x, y=y)

                with st.expander("Line Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="line_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="line_y")
                    st.info(f"Plotting **{y}** vs **{x}**")
                    lineplot(df_to_use, x=x, y=y)

        # Master toggle for 'Data Preprocessing'
        if st.toggle("🔧 Enable Data Preprocessing"):
            
            st.subheader("🧹 Data Preprocessing Tools")

            # ==========================================================
            # 1) CLEAN MISSING VALUES
            # ==========================================================
            with st.expander("🧼 Clean Missing Values"):
                missing_count = df_to_use.isna().sum().sum()
                st.write(f"Missing values detected: **{missing_count}**")

                strategy = st.radio(
                    "Choose strategy:",
                    ["None", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill Custom Value"],
                    index=0
                )

                if strategy == "Drop Rows":
                    df_to_use.dropna(inplace=True)
                    st.success("✔ Rows with missing values removed.")

                elif strategy == "Drop Columns":
                    df_to_use.dropna(axis=1, inplace=True)
                    st.success("✔ Columns containing missing values removed.")

                elif strategy == "Fill with Mean":
                    df_to_use.fillna(df_to_use.mean(numeric_only=True), inplace=True)
                    st.success("✔ Missing numerical values filled using column means.")

                elif strategy == "Fill with Median":
                    df_to_use.fillna(df_to_use.median(numeric_only=True), inplace=True)
                    st.success("✔ Missing numerical values filled using medians.")

                elif strategy == "Fill with Mode":
                    df_to_use.fillna(df_to_use.mode().iloc[0], inplace=True)
                    st.success("✔ Missing values filled using mode.")

                elif strategy == "Fill Custom Value":
                    custom = st.text_input("Enter value to fill missing cells:")
                    if custom:
                        df_to_use.fillna(custom, inplace=True)
                        st.success(f"✔ Missing values replaced with **{custom}**")


            # ==========================================================
            # 2) REMOVE DUPLICATES
            # ==========================================================
            with st.expander("🗃️ Remove Duplicates"):
                duplicates = df_to_use.duplicated().sum()
                st.write(f"Duplicate rows detected: **{duplicates}**")

                if st.button("Remove Duplicates Now"):
                    df_to_use.drop_duplicates(inplace=True)
                    st.success("✔ Duplicate records removed successfully.")


            # ==========================================================
            # 3) HANDLE OUTLIERS (IQR Method)
            # ==========================================================
            with st.expander("🔍 Handle Outliers"):
                num_cols = df_to_use.select_dtypes(include=["int", "float"]).columns.tolist()

                if len(num_cols) == 0:
                    st.warning("⚠ No numeric columns available for outlier detection.")
                else:
                    col = st.selectbox("Select column to evaluate:", num_cols)

                    if st.button(f"Apply IQR Outlier Filtering on `{col}`"):
                        Q1 = df_to_use[col].quantile(0.25)
                        Q3 = df_to_use[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

                        before = len(df_to_use)
                        df_to_use = df_to_use[(df_to_use[col] >= lower) & (df_to_use[col] <= upper)]
                        removed = before - len(df_to_use)

                        st.success(f"✔ Outliers removed from **{col}** | Rows dropped: **{removed}**")


            # ==========================================================
            # 4) DATA NORMALIZATION / SCALING
            # ==========================================================
            with st.expander("⚖️ Data Normalization"):
                scale_method = st.radio(
                    "Choose scaling method:",
                    ["None", "Min-Max Scaling (0→1)", "Standard Scaling (Z-score)"]
                )

                num_cols = df_to_use.select_dtypes(include=["int", "float"]).columns.tolist()

                if scale_method != "None" and len(num_cols) > 0:

                    from sklearn.preprocessing import MinMaxScaler, StandardScaler

                    if scale_method == "Min-Max Scaling (0→1)":
                        scaler = MinMaxScaler()
                        df_to_use[num_cols] = scaler.fit_transform(df_to_use[num_cols])
                        st.success("✔ Feature scaling completed (Min-Max).")

                    elif scale_method == "Standard Scaling (Z-score)":
                        scaler = StandardScaler()
                        df_to_use[num_cols] = scaler.fit_transform(df_to_use[num_cols])
                        st.success("✔ Standard normalization applied (mean=0, std=1).")

                elif scale_method != "None":
                    st.warning("⚠ No numeric features available for scaling.")

            st.success("✨ Preprocessing applied! You may now proceed to model training.")


with train_ml_model:
    st.info(
        "Here, you can train machine learning models on the selected dataset, "
        "configure the hyperparameters, and monitor the training process to achieve "
        "the best performance."
    )
    if not df_to_use.empty:

        st.subheader("🌲 Train CART Decision Tree Model")

        # Model type classification/regression
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
            max_depth = st.number_input("Max Depth (None = auto)", min_value=1, value=5)
        with col2:
            min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2)
        with col3:
            test_size = st.slider("Test Size %", 5, 40, 20) / 100

        criterion = "gini" if model_type == "classification" else "squared_error"

        # Train button
        if st.button("🚀 Train Decision Tree Model"):
            from src.model import train_cart_decision_tree, save_model
            import pandas as pd
            from sklearn.preprocessing import LabelEncoder

            # Prepare features
            X = df_to_use.drop(columns=[target_column]).copy()
            y = df_to_use[target_column].copy()

            # Convert non-numeric columns in features
            for col in X.select_dtypes(include='object').columns:
                try:
                    X[col] = pd.to_datetime(X[col], format="%m-%Y").map(lambda x: x.timestamp())
                except:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])

            # Encode target if classification and non-numeric
            if model_type == "classification" and y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            # For regression, convert month-year strings to timestamps
            if model_type == "regression" and y.dtype == "object":
                try:
                    y = pd.to_datetime(y, format="%m-%Y").map(lambda x: x.timestamp())
                except:
                    st.error("Selected target column is non-numeric. Please choose a numeric column for regression.")
                    st.stop()

            # Train model
            model, metrics = train_cart_decision_tree(
                df=df_to_use,
                target_column=target_column,
                model_type=model_type,
                test_size=test_size,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion
            )

            st.success("🎉 Model trained successfully!")
            st.info(f"📈 Performance → {metrics}")

            # Store trained model for prediction tab
            st.session_state["trained_model"] = model
            st.session_state["trained_features"] = X.columns.tolist()
            st.session_state["trained_target"] = target_column


            if st.button("💾 Save Model"):
                path = save_model(model)
                st.success(f"Model saved to: `{path}`")


with prediction_tab:
    st.title("🔮 Model Prediction")
    st.info("Use your trained model to make predictions and generate forecasts.")

    # ============================================
    # 1) Check if trained model exists
    # ============================================
    if (
        "trained_model" not in st.session_state
        or "trained_features" not in st.session_state
        or "trained_target" not in st.session_state
    ):
        st.error("❌ No trained model found. Please train a model in the previous tab first.")
        st.stop()

    model = st.session_state["trained_model"]
    trained_features = st.session_state["trained_features"]
    trained_target = st.session_state["trained_target"]

    st.success(f"📌 Model will predict the target column: **{trained_target}**")

    # ============================================
    # 2) Upload dataset for prediction
    # ============================================
    st.subheader("📁 Upload Dataset for Prediction")
    pred_file = st.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])

    if pred_file:
        df_pred = pd.read_excel(pred_file)
        st.write("### 👀 Preview of Uploaded Data")
        st.dataframe(df_pred.head())
    else:
        st.warning("Please upload a dataset to perform predictions.")
        st.stop()

    # ============================================
    # 3) Ensure required columns exist
    # ============================================
    missing_cols = set(trained_features) - set(df_pred.columns)
    if missing_cols:
        st.error(f"❌ Missing required feature columns: {missing_cols}")
        st.stop()

    # ============================================
    # 4) Preprocess for prediction (same as training)
    # ============================================
    st.subheader("⚙ Auto Preprocessing for Prediction")

    df_input = df_pred[trained_features].copy()

    from sklearn.preprocessing import LabelEncoder

    for col in df_input.columns:

        # Try: datetime → timestamp
        try:
            df_input[col] = pd.to_datetime(df_input[col], errors="raise")
            df_input[col] = df_input[col].astype("int64") // 10**9
            continue
        except:
            pass

        # Encode non-numeric
        if df_input[col].dtype == "object":
            le = LabelEncoder()
            df_input[col] = le.fit_transform(df_input[col].astype(str))

    # Safety: convert any remaining datetime64
    for col in df_input.columns:
        if str(df_input[col].dtype).startswith("datetime64"):
            df_input[col] = df_input[col].astype("int64") // 10**9

    # ============================================
    # 5) Predict
    # ============================================
    if st.button("🔮 Run Prediction"):
        try:
            preds = model.predict(df_input)

            df_result = df_pred.copy()
            df_result[trained_target] = preds  # add predicted column

            st.success("🎉 Prediction Completed Successfully!")
            st.write("### 📊 Prediction Output")
            st.dataframe(df_result)

            # Download
            output_path = "prediction_results.xlsx"
            df_result.to_excel(output_path, index=False)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="📥 Download Predictions",
                    data=f,
                    file_name="prediction_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
