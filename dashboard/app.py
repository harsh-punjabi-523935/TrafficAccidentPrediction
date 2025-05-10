import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load data
df = pd.read_csv("/data/processed/processed_data.csv")

st.dataframe(df.head())

st.title("🚦 Toronto Traffic Accident Dashboard")
st.markdown("Explore collision patterns and predict accident impact types.")

# Filters
st.sidebar.header("Filter by:")
selected_day = st.sidebar.multiselect("Day of Week", df['WEEKDAY'].unique(), default=df['WEEKDAY'].unique())
selected_month = st.sidebar.multiselect("Month", df['MONTH'].unique(), default=df['MONTH'].unique())

filtered_df = df[(df['WEEKDAY'].isin(selected_day)) & (df['MONTH'].isin(selected_month))]

# Impact Type Distribution
st.subheader("Impact Type Distribution")
impact_counts = filtered_df['IMPACTYPE'].value_counts()
st.bar_chart(impact_counts)

# Top 10 Accident Locations
st.subheader("Top 10 Accident Locations")
top_locations = filtered_df['ACCLOC'].value_counts().nlargest(10)
st.bar_chart(top_locations)

# Optional Prediction Section
st.subheader("📊 Predict Impact Type (optional)")

with st.form("predict_form"):
    year = st.selectbox("Year", sorted(df['YEAR'].unique()))
    month = st.selectbox("Month", sorted(df['MONTH'].unique()))
    day = st.selectbox("Day", sorted(df['DAY'].unique()))
    weekday = st.selectbox("Weekday", df['WEEKDAY'].unique())
    accloc = st.selectbox("Accident Location", df['ACCLOC'].unique())
    submit = st.form_submit_button("Predict")

    if submit:
        # Load label encoders and model
        model = joblib.load("../models/rf_model.pkl")
        le_weekday = joblib.load("../models/le_weekday.pkl")
        le_accloc = joblib.load("../models/le_accloc.pkl")
        le_impact = joblib.load("../models/le_impact.pkl")

        input_df = pd.DataFrame({
            "YEAR": [year],
            "MONTH": [month],
            "DAY": [day],
            "WEEKDAY": le_weekday.transform([weekday]),
            "ACCLOC": le_accloc.transform([accloc])
        })

        pred = model.predict(input_df)
        predicted_label = le_impact.inverse_transform(pred)
        st.success(f"Predicted Impact Type: **{predicted_label[0]}**")

st.download_button("Download Filtered Data", filtered_df.to_csv(index=False), "filtered_data.csv", "text/csv")

