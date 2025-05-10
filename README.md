# 🚦 Toronto Traffic Accident Prediction Dashboard

This project analyzes motor vehicle collision data in Toronto and predicts the type of traffic impact based on date, weekday, and location. It includes data cleaning, exploratory analysis, predictive modeling, and an interactive Streamlit dashboard.

---

## 🔍 Features

- 📊 Visualize collision trends by time and location
- 🧠 Predict the type of collision impact (e.g., Rear End, Turning)
- 🎛️ Filter data by weekday and month
- 📥 Download filtered datasets
- 🌍 Optional: map-based heatmap of accident locations

---

## 🛠 Tech Stack

- Python 3.11
- Streamlit
- Pandas, Seaborn, Matplotlib
- scikit-learn
- Joblib

---

## 📂 Folder Structure

notebooks/ → Jupyter Notebooks for EDA and Modeling
data/processed/ → Cleaned dataset for use in dashboard
models/ → Trained model and encoders (.pkl files)
dashboard/app.py → Main Streamlit application
requirements.txt → Python dependencies
README.md → Project overview

## 🚀 Run the Dashboard Locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
