# Standard Library
import os
import sys

# Add the project root to sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Third-Party Libraries
import pandas as pd
import streamlit as st

from strings.strings import (
    PLATFORM_DESCRIPTION,
    UPLOAD_INSTRUCTION,
    WELCOME_MESSAGE,
    WORKFLOW_SUPPORT_MESSAGE,
)

# Local Imports — App Tabs
from app.prediction_tab import run_prediction_tab
from app.train_tab import render_train_tab
from app.upload_tab import run_upload_tab
from app.data_eng_tab import run_data_eng_tab

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
    "🔮 Prediction",
])

with home:
    st.title("Welcome to Well Production Forecasting Dashboard 🛢️📈")
    st.write(WELCOME_MESSAGE)
    st.write(PLATFORM_DESCRIPTION)
    st.write(UPLOAD_INSTRUCTION)
    st.write(WORKFLOW_SUPPORT_MESSAGE)

with upload_dataset:
    run_upload_tab()

with data_eng_tab:
    run_data_eng_tab()

with train_ml_model:
    df_to_use = st.session_state.get("df_to_use")

    if df_to_use is not None:
        render_train_tab(df_to_use)
    else:
        st.warning("⚠️ No dataset found. Please complete the Data Engineering tab first.")

with prediction_tab:
    run_prediction_tab()
