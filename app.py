import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")

st.title("🌾 Crop Yield Prediction")
st.markdown("Enter the details below to predict crop yield")

# Load unique values for dropdowns
import pandas as pd
df = pd.read_csv("Data/clean_df.csv")
countries = sorted(df['Area'].unique().tolist())
crops = sorted(df['Item'].unique().tolist())

# Input fields
col1, col2 = st.columns(2)

with col1:
    Area = st.selectbox("Country", countries)
    Year = st.number_input("Year", min_value=1990, max_value=2030, value=2010)
    avg_temp = st.number_input("Average Temperature (°C)", value=20.0)

with col2:
    Item = st.selectbox("Crop Type", crops)
    average_rain_fall_mm_per_year = st.number_input("Rainfall (mm/year)", value=1000.0)
    pesticides_tonnes = st.number_input("Pesticides (tonnes)", value=100.0)

# Predict button
if st.button("Predict Yield 🌾"):
    data = CustomData(
        Area=Area,
        Item=Item,
        Year=Year,
        average_rain_fall_mm_per_year=average_rain_fall_mm_per_year,
        pesticides_tonnes=pesticides_tonnes,
        avg_temp=avg_temp
    )

    pred_df = data.get_data_as_data_frame()
    pipeline = PredictPipeline()
    result = pipeline.predict(pred_df)

    st.success(f"Predicted Crop Yield: **{result[0]:,.0f} hg/ha**")
    st.info(f"That is approximately **{result[0]/10000:.2f} tonnes/hectare**")