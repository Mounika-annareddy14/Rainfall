# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:00:20 2024

@author: Mounika Reddy
"""


import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Load the saved model
model_path = 'G:/My Drive/Model_deploy/rainfall_predict.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))

# Add custom CSS for background
st.markdown(
    """
    <style>
    body {
        background-image: url('G:\My Drive\Model_deploy\rainfall.jpeg'); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main > div {
        background-color: rgba(255, 255, 255, 0.9); /* Add transparency */
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.title("RainFall Prediction App")

# Display an image
image_path = "http://wallpapercave.com/wp/u5T2tgd.jpg"
try:
    st.image(image_path, use_container_width=True)
except FileNotFoundError:
    st.warning("Image not found. Please check the file path or upload the image.")

# Display the first 5 rows of the dataset
st.subheader("Sample Dataset Used for Model Training")
df = pd.read_csv("G:\My Drive\Model_deploy\Rainfall.p.csv")
st.dataframe(df.head(5))  # Display first 5 rows of the dataset

# Prediction function
def rainfall_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "Rain has no chance"
    else:
        return "It is raining"

# Main function
def main():
    st.subheader("Enter Climate Details for Prediction")

    # Use two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        pressure = st.number_input("Pressure of Climate", min_value=950.0, max_value=1050.0, value=1015.0, step=0.1)
        max_temp = st.number_input("Max Temperature of Climate", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
        humidity = st.number_input("Humidity of Climate", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        wind_speed = st.number_input("Wind Speed of Climate", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        sunshine = st.number_input("Sunshine of Climate", min_value=0.0, max_value=12.0, value=6.0, step=0.1)

        # Adding the selectbox for Weather Condition in the first column
        weather_condition = st.selectbox("Select Weather Condition", options=["Stormy", "Rainy"])

    with col2:
        min_temp = st.number_input("Min Temperature of Climate", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
        dew_point = st.number_input("Dew Point of Climate", min_value=-10.0, max_value=30.0, value=15.0, step=0.1)
        cloud = st.number_input("Cloud Status of Climate", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        wind_direction = st.number_input("Wind Direction of Climate", min_value=0.0, max_value=360.0, value=180.0, step=1.0)

    # Convert weather condition to label encoding (1 for Stormy, 0 for Rainy)
    weather_encoded = 1 if weather_condition == "Stormy" else 0

    # Calculate temperature range
    temp_range = max_temp - min_temp
    st.write(f"The calculated temperature range is: {temp_range:.2f}")

    # Button for prediction
    if st.button("Predict Rainfall"):
        # Including the encoded weather condition as an additional feature in the input data
        input_data = [pressure, max_temp, min_temp, temp_range, dew_point, humidity, cloud, sunshine, wind_direction, wind_speed, weather_encoded]
        result = rainfall_prediction(input_data)
        st.success(f"Prediction: {result}")
        st.write(f"Selected Weather Condition: {weather_condition}")  # Display the selected weather condition

# Run the app
if __name__ == "__main__":
    main()
