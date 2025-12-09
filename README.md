# 🛢️ Well Production Forecasting Dashboard

A Streamlit-based machine learning dashboard for **visualizing**, **cleaning**, **engineering**, **training**, and **forecasting** oil & gas well production.

![License](https://img.shields.io/github/license/sobhankohanpour/production-forecasting-dashboard)
![Last Commit](https://img.shields.io/github/last-commit/sobhankohanpour/production-forecasting-dashboard)
![Issues](https://img.shields.io/github/issues/sobhankohanpour/production-forecasting-dashboard)
![Pull Requests](https://img.shields.io/github/issues-pr/sobhankohanpour/production-forecasting-dashboard)
![Repo Size](https://img.shields.io/github/repo-size/sobhankohanpour/production-forecasting-dashboard)
![Code Size](https://img.shields.io/github/languages/code-size/sobhankohanpour/production-forecasting-dashboard)
![Contributors](https://img.shields.io/github/contributors/sobhankohanpour/production-forecasting-dashboard)
![Forks](https://img.shields.io/github/forks/sobhankohanpour/production-forecasting-dashboard)
![GitHub Stars](https://img.shields.io/github/stars/sobhankohanpour/production-forecasting-dashboard)


## 🚀 Overview

The **Well Production Forecasting Dashboard** is an end-to-end machine learning application designed for petroleum engineers, reservoir analysts, and data scientists. It enables you to:

* Import custom datasets or use included real-world samples
* Visualize production trends with interactive plots
* Prepare and process data for analysis and modeling
* Build CART machine-learning models
* Produce accurate well production forecasts

The app is organized into five interactive Streamlit tabs, offering a smooth and guided workflow from raw data to final prediction.


## 📁 Project Structure

```
production-forecasting-dashboard/
│
├── app/
│   ├── main.py               # Main Streamlit app with all tabs
│   ├── upload_tab.py         # Dataset upload + built-in dataset selector
│   ├── data_eng_tab.py       # Visualization + preprocessing tools
│   ├── train_tab.py          # CART model training + evaluation
│   ├── prediction_tab.py     # Prediction using trained model
│
├── data/                     # Included real-world ND datasets
│   ├── ND_cumulative_formation_2020.xlsx
│   ├── ND_gas_1990_to_present.xlsx
│   ├── ND_historical_barrels_of_oil_produced_by_county.xlsx
│   └── ND_historical_MCF_gas_produced_by_county.xlsx
│
├── src/
│   ├── plots.py              # Unified Seaborn + Matplotlib plotting utilities
│   └── model.py              # CART model builder, evaluator, saver
│
├── strings/
│   └── strings.py            # UI messages and text constants
│
├── README.md
├── requirements.txt
└── LICENSE
```


## 🧠 Features

### 🔹 1. Dataset Handling

* Upload custom **`.xlsx` files**
* Select from **four included North Dakota datasets**
* Automatic dataset summary and preview
* Preprocessing support for modeling and visualization


### 🔹 2. Exploratory Data Analysis

The dashboard includes **15+ interactive plot types**, grouped into:

#### 📊 Distribution Plots

* Distribution plot
* Histogram (configurable bins)
* KDE
* ECDF
* Rug plot

#### 🧩 Categorical Plots

* Catplot
* Strip plot
* Swarm plot
* Box plot
* Violin plot
* Point plot
* Bar plot

#### 🔗 Relational Plots

* Scatter plot
* Line plot

All plots use clean Seaborn + Matplotlib visuals optimized for Streamlit.


## 🤖 Machine Learning (CART)

The app supports **CART decision tree models** for both regression and classification.

### ✔ Automatically handles:

* Numeric columns
* Date/time formatting
* Label encoding for non-numeric features

### ✔ Model evaluation includes:

* **Accuracy** (classification)
* **MSE** & **R²** (regression)

Models can be trained, evaluated, and saved locally for later predictions.


## 🔮 Prediction

* Generate predictions using trained CART models
* Interactive input forms
* Downloadable prediction results


## ▶️ How to Run the App

### **1️⃣ Clone the repository**

```bash
git clone https://github.com/sobhankohanpour/production-forecasting-dashboard.git
cd production-forecasting-dashboard
```

### **2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Launch the Streamlit app**

```bash
streamlit run app/main.py
```


## 📦 Dependencies

Requires Python **3.8+**

Core libraries:

* streamlit
* pandas
* matplotlib
* seaborn
* scikit-learn
* joblib


## 📘 Included Datasets

The `data/` directory includes curated North Dakota production datasets:

* **Cumulative Oil Production by Formation (2020)**
* **Gas Production (1990–Present)**
* **Historical Monthly Oil Production by County**
* **Historical Monthly Gas Production by County**

All datasets are directly accessible from within the dashboard.


## 🧩 Plot Utilities (`src/plots.py`)

Reusable plotting functions include:

* `scatterplot()`
* `lineplot()`
* `distplot()`, `histplot()`, `kdeplot()`
* `ecdfplot()`, `rugplot()`
* `catplot()`, `stripplot()`, `swarmplot()`
* `boxplot()`, `violinplot()`
* `pointplot()`, `barplot()`

Easy to extend for custom visualizations.


## 📄 License

MIT License — free for personal and commercial use.


## 🤝 Contributing

Contributions, enhancements, and feature requests are welcome!
Feel free to open an issue or submit a pull request.


## ⭐ Support

If you find this project useful, please consider giving it a **star** ⭐ on GitHub.
