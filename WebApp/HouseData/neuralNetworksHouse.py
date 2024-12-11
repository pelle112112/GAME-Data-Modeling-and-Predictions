import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras import models

# Load pre-trained data and model
with open('Data/HouseData/Neural.pkl', 'rb') as file:
    data = pickle.load(file)

# Load the data and label encoders


df = data['dataframe']
model = data['model']
scaler = data['scaler']

# Streamlit app function
def neuralNetworksPredictions():
    st.title("Neural Network Predictions House Data")
    st.write("### This is a neural network model that predicts the number of attendees in a given GAME House based on the input features")
    
    st.write("Model prediction accuracy and performance numbers:")
    mse= 17383.38970995191
    mae= 104.05304445266961
    r2= 0.3895542025566101
    st.write("Mean Squared Error: ", mse)
    st.write("Mean Absolute Error: ", mae)
    st.write("R^2 Score: ", r2)

    municipalityCodesnew = {
        "København": 101,
        "Aalborg": 851,
        "Esbjerg": 561,
        "Viborg": 791
    }
    

    # Lets create an array of month objects which holds the min and max temperature for each month
    month_temps = [
        (-31, 13, 2),  # January
        (-29, 16, 3),  # February
        (-27, 23, 5),  # March
        (-19, 29, 10),  # April
        (-8, 33, 16),  # May
        (-4, 36, 20),  # June
        (-1, 36, 20),  # July
        (-2, 35, 20),  # August
        (-6, 33, 20),  # September
        (-12, 27, 13),  # October
        (-22, 19, 8),  # November
        (-26, 15, 4)  # December
    ]

    

    selectDateOrNot = st.checkbox("Select Date")
    if(selectDateOrNot):
        date = st.date_input("Date")
        dayOfweekselection = 0
        updatedDay = date.weekday()
        month = date.month
        st.write("Day of the week: ", updatedDay)
        st.write("Month: ", month)
        st.write("Date: ", date)
    else:
        st.write("Date: Not Selected")
        dayOfweekselection = st.selectbox("Day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        month = st.slider("Month", 1, 12, 1)
    
    # Select city and extract municipality code
    municipalityCodeSelection = st.selectbox("Municipality Code", list(municipalityCodesnew.keys()))
    municipalityCode = municipalityCodesnew[municipalityCodeSelection]
   
    temperature = st.slider("Temperature", month_temps[month - 1][0], month_temps[month - 1][1], month_temps[month - 1][2])
    max_mean_temp = st.slider("Max Mean Temperature", month_temps[month - 1][0], month_temps[month - 1][1], month_temps[month - 1][2])
    holiday = st.selectbox("Holiday", ["Yes", "No"])
    rain = st.slider("Rain", 0, 3, 0)
    
    ok = st.button("Predict")

    if ok:
        # Preprocess input
        if(dayOfweekselection == 0):
            dayOfweek = updatedDay
        else:
            dayOfweek = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}[dayOfweekselection]
  
        holiday = 1 if holiday == "Yes" else 0
        municipalityCode = 101  # Default value
        municipalityCode = municipalityCodesnew[municipalityCodeSelection]
        rainValue = rain

        input_data1 = np.array([[holiday, month, dayOfweek]])
        input_data2 = scaler.fit_transform(np.array([[max_mean_temp, rainValue, temperature]]))
        
        input_data = np.concatenate((input_data2, input_data1), axis=1)


        # Make prediction
        prediction = model.predict(input_data)
        st.write("Municipality Code: ", municipalityCode)
        st.write(f"Predicted number of attendees: {int(prediction[0][0])}")

# Call the function to make predictions
neuralNetworksPredictions()