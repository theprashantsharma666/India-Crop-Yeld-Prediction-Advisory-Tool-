# 🌾 India Crop Yield Prediction & Advisory Tool

An end-to-end machine learning web application that predicts crop yield
for 20+ Indian crops across 15 states and provides actionable farming
advisory — built with Python, Scikit-learn, XGBoost, and Streamlit.

---

## 📌 Project Overview

Farmers in India often lack access to data-driven tools that can help
them make informed decisions about which crop to grow based on their
soil, region, and season. This project addresses that gap by building
a complete ML pipeline — from raw data to a deployed interactive web
application — that predicts crop yield and advises farmers accordingly.

The tool takes inputs like state, district, crop type, season, and area
under cultivation, and returns a predicted yield along with comparative
insights across similar conditions.

---

## ✨ Features

- 🔍 **Crop Yield Prediction** — Predicts yield (tonnes/hectare) for
  20+ crops across 15 Indian states
- 🤖 **Dual Model Comparison** — Trains and evaluates both Random Forest
  and XGBoost; selects the best model using RMSE and R²
- 📊 **Exploratory Data Analysis (EDA)** — Interactive visualizations
  showing yield trends, state-wise distribution, and seasonal patterns
- 🧹 **Automated Preprocessing** — Handles missing values, encodes
  categorical features, and scales numerical inputs in a reusable pipeline
- 💡 **Farming Advisory** — Provides crop-specific recommendations based
  on predicted yield range
- 🌐 **Streamlit Web App** — Clean, interactive UI accessible directly
  in the browser with no local setup required

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Model Evaluation | RMSE, R², Cross-validation |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Dataset Sources | Kaggle (soil data), FAO (yield data) |
| Version Control | Git, GitHub |

---

## 📂 Project Structure
india-crop-yield-prediction/
│
├── app.py                  # Streamlit app — UI, user inputs, displays prediction
├── main.py                 # Trains models, evaluates RMSE & R², saves best model
├── preprocessor.py         # Data cleaning, encoding, scaling pipeline
├── helper.py               # Utility functions — load model, predict, format output
├── eda.py                  # EDA logic — summary stats, distributions, correlations
├── visualization.py        # All charts & plots — yield trends, state-wise, seasonal
│
├── data/
│   ├── crop_yield.csv      # FAO yield dataset (state, crop, year, production, area)
│   └── soil_data.csv       # Kaggle soil dataset (N, P, K, pH by region)
│
├── models/
│   └── best_model.pkl      # Saved best model (Random Forest or XGBoost)
│
├── requirements.txt        # All dependencies (pandas, scikit-learn, xgboost, streamlit)
├── .gitignore              # Ignore venv/, __pycache__/, *.pyc, .env
└── README.md

---

## 📊 Dataset Information

| Dataset | Source | Description |
|---|---|---|
| Crop Yield Data | FAO (Food & Agriculture Organization) | Year-wise yield records for Indian crops by state and season |
| Soil Data | Kaggle | Soil nutrient composition (N, P, K, pH) by region |

**Key columns used:**
- `State_Name` — Indian state
- `District_Name` — District within state
- `Crop_Year` — Year of cultivation
- `Season` — Kharif / Rabi / Whole Year
- `Crop` — Crop name (Rice, Wheat, Maize, etc.)
- `Area` — Area under cultivation (hectares)
- `Production` — Actual production (tonnes) → used to derive yield

**Target variable:** `Yield = Production / Area` (tonnes/hectare)

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/theprashantsharma666/india-crop-yield-prediction.git
cd india-crop-yield-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. Open in browser

---

## 📈 Model Performance

| Model | RMSE | R² Score |
|---|---|---|
| Random Forest | — | — |
| XGBoost | — | — |
| **Best Model** | **—** | **—** |

> Fill in your actual RMSE and R² values after training.
> Best model is saved automatically to `models/best_model.pkl`

---

## 🚀 Future Improvements

- [ ] Add real-time weather API integration (temperature, rainfall)
      to improve prediction accuracy
- [ ] Expand dataset to cover all 28 Indian states
- [ ] Include soil nutrient input fields (N, P, K, pH) in the UI
- [ ] Add crop recommendation feature — suggest best crop for given
      conditions rather than just predicting yield
- [ ] Deploy on Streamlit Cloud or Render for public access
- [ ] Add multilingual support (Hindi) for broader farmer accessibility
- [ ] Integrate MLflow for experiment tracking and model versioning

---

## 👨‍💻 Author

**Prashant Sharma**
B.Tech CSE (AI & Data Science) — GLA University, Mathura
📧 prashant.sharma_cs24@gla.ac.in
🔗 [LinkedIn](https://linkedin.com/in/prashant-sharma-52492b330)
🐙 [GitHub](https://github.com/theprashantsharma666)

---

## 📄 License

This project is under the [MIT License](LICENSE).
