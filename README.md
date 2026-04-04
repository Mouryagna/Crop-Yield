# 🌾 Crop Yield Prediction System

A machine learning system that predicts crop yield (hg/ha) based on environmental and agricultural factors, built with XGBoost and deployed via Streamlit.

---

## 📌 Overview

Crop yield prediction is a critical problem in modern agriculture. This project builds an end-to-end ML pipeline that takes environmental inputs such as rainfall, temperature, and pesticide usage to predict the expected crop yield for a given country and crop type.

The model achieves an **R² score of 0.98** on the test set using XGBoost with GridSearchCV hyperparameter tuning.

---

## 📂 Project Structure

```
Crop-Yield/
│
├── Data/
│   └── clean_df.csv
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── ml_train.ipynb
│   └── LSTM_train.ipynb
│
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   ├── raw.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── encoder.pkl
│
├── src/
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py
│       └── predict_pipeline.py
│
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📊 Dataset

The dataset is sourced from **FAO (Food and Agriculture Organization)** via Kaggle and consists of 5 CSV files:

| File | Description |
|---|---|
| `yield.csv` | Raw crop yield data by country and year |
| `rainfall.csv` | Average rainfall (mm/year) by country |
| `temp.csv` | Average temperature by country and year |
| `Pesticides.csv` | Pesticide usage (tonnes) by country |
| `yield_df.csv` | Pre-merged version of all above files |

**Final merged dataset:** ~28,000 rows across 101 countries and 10 crop types (1990–2013)

---

## 🧪 Features Used

| Feature | Type | Description |
|---|---|---|
| `Area` | Categorical | Country name |
| `Item` | Categorical | Crop type |
| `Year` | Numerical | Year of record |
| `average_rain_fall_mm_per_year` | Numerical | Annual rainfall in mm |
| `pesticides_tonnes` | Numerical | Pesticide usage in tonnes |
| `avg_temp` | Numerical | Average temperature in °C |

### ⚙️ Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| `hg/ha_yield_log` | `log1p(yield)` | Fix target skewness |
| `pesticides_cbrt` | `cbrt(pesticides)` | Fix feature skewness |
| `temp_stress` | `abs(temp - 20)` | Capture temperature deviation from optimal |
| `rain_temp_ratio` | `rainfall / (temp + 1)` | Interaction between rainfall and temperature |
| `year_trend` | `year - 1990` | Capture technology improvement over time |
| `rainfall_category` | Binned rainfall | Ordinal encoding of rainfall levels |
| `temp_category` | Binned temperature | Ordinal encoding of temperature levels |

---

## 🤖 Model

| Stage | Details |
|---|---|
| Algorithm | XGBoost Regressor |
| Tuning | GridSearchCV (cv=5) |
| Encoding | Target Encoding (Area, Item) |
| Scaling | StandardScaler |
| Target | `log1p(hg/ha_yield)` → reversed with `expm1` |

### Best Hyperparameters

```
learning_rate: 0.2
max_depth: 7
n_estimators: 200
subsample: 1.0
```

### Results

| Metric | Value |
|---|---|
| R² Score | 0.98 |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Mouryagna/Crop-Yield.git
cd Crop-Yield
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Train the model
```bash
python -m src.components.data_ingestion
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🌐 App

The Streamlit app takes 6 user inputs and predicts crop yield in the background:

**User Inputs:**
- Country
- Crop Type
- Year
- Rainfall (mm/year)
- Average Temperature (°C)
- Pesticides (tonnes)

**Output:**
- Predicted yield in **hg/ha**
- Converted yield in **tonnes/hectare**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Preprocessing, pipelines |
| XGBoost | ML model |
| Category Encoders | Target encoding |
| Streamlit | Web app deployment |
| Matplotlib, Seaborn | EDA visualizations |
| Dill | Model serialization |

---

## 👤 Author

**Mouryagna Baindla**
- GitHub: [@Mouryagna](https://github.com/Mouryagna)
- Email: mouryagnabaindla@gmail.com