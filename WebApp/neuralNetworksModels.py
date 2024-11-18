import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras import models

# Load pre-trained data and model
def model_load():
    with open('../Data/Neural.pkl', 'rb') as file:
        data = pickle.load(file)
    # Assuming that the 'labels' file is available and contains label encoders
    with open('../Data/EventData.pkl', 'rb') as file2:
        labels = pickle.load(file2)
    return data, labels

# Load the data and label encoders
data, labels = model_load()
df = data['dataframe']
scaler = data['scaler']
model = models.load_model('../Data/NeuralModel.h5')

# Extract label encoders
le_EventTypeLoaded = labels['le_EventType']
le_ZonesLoaded = labels['le_Zones']
# If using other label encoders, make sure they're loaded similarly

# Streamlit app function
def neuralNetworksPredictions():
    st.title("Neural Network Predictions")

    eventTypes = (
        'Street Football',
        'GAME Girl Zone',
        'Street Basketball',
        'Street Dance',
        'Street Dance, Street Basketball, Street Football, GAME Girl Zone, Skate, Skateboarding',
        'Skate',
        'Street Basketball, Street Football')
    

    zones = (
        'Hørgården - København',
        'Mjølnerparken - København',
        'Den Grønne Trekant - Østerbro',
        'Other Zones',
        'Søndermarken - Frederiksberg',
        'Sydbyen - Næstved',
        'GAME Streetmekka Viborg',
        'Rosenhøj/Viby - Aarhus (Street soccer)',
        'Gellerup/Brabrand - Aarhus',
        'Fri-Stedet - Aalborg',
        'Herlev',
        'Stengårdsvej - Esbjerg',
        'Munkevænget - Kolding',
        'Stjernen - Frederiksberg',
        'Kalbyris - Næstved',
        'Nordvest - Tagensbo',
        'Odense - Ejerslykke',
        'Platformen -Esbjerg',
        'Rosenhøj/Viby - Aarhus (GGZ)',
        'Skovparken-Kolding',
        'Nørrebro - Rådmandsgade Skole',
        'Frydenlund-Aarhus',
        'TK Ungdomsgård',
        'Aarhus Nord',
        'Spektrumparken - Esbjerg',
        'Aalborg Øst',
        'Stensbjergparken - Sønderborg')
    
    zone_municipality_codes = {
    'Hørgården - København': 101,
    'Mjølnerparken - København': 101,
    'Den Grønne Trekant - Østerbro': 101,
    'Other Zones': 461,  #Used Odenese as the default municipality code
    'Søndermarken - Frederiksberg': 101,
    'Sydbyen - Næstved': 370,
    'GAME Streetmekka Viborg': 791,
    'Rosenhøj/Viby - Aarhus (Street soccer)': 751,
    'Gellerup/Brabrand - Aarhus': 751,
    'Fri-Stedet - Aalborg': 851,
    'Herlev': 163,
    'Stengårdsvej - Esbjerg': 561,
    'Munkevænget - Kolding': 621,
    'Stjernen - Frederiksberg': 101,
    'Kalbyris - Næstved': 370,
    'Nordvest - Tagensbo': 101,
    'Odense - Ejerslykke': 461,
    'Platformen -Esbjerg': 561,
    'Rosenhøj/Viby - Aarhus (GGZ)': 751,
    'Skovparken-Kolding': 621,
    'Nørrebro - Rådmandsgade Skole': 101,
    'Frydenlund-Aarhus': 751,
    'TK Ungdomsgård': 370,
    'Aarhus Nord': 751,
    'Spektrumparken - Esbjerg': 561,
    'Aalborg Øst': 851,
    'Stensbjergparken - Sønderborg': 540
    }

    

    dayOfweekselection = st.selectbox("Day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    eventType = st.selectbox("Event Type", eventTypes)
    zone = st.selectbox("Zone", zones)
    zonename = zone
    month = st.slider("Month", 1, 12, 1)
    temperature = st.slider("Temperature", -10, 40, 20)
    max_mean_temp = st.slider("Max Mean Temperature", -10, 40, 20)
    holiday = st.selectbox("Holiday", ["Yes", "No"])
    attendin = st.slider("Attendin", 0, 100, 50)

    ok = st.button("Predict")

    if ok:
        # Preprocess input
        dayOfweek = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}[dayOfweekselection]
        zone = le_ZonesLoaded.transform([zone])[0]
        eventType = le_EventTypeLoaded.transform([eventType])[0]
        holiday = 1 if holiday == "Yes" else 0
        municipalityCode = 461  # Default value
        for zonename in zone_municipality_codes:
            if zonename == zone:
                municipalityCode = zone_municipality_codes[zonename]
                break

        input_data = np.array([[dayOfweek, eventType, zone, month, temperature, holiday, municipalityCode]])
        input_data = np.array([[eventType, zone, municipalityCode, dayOfweek, month, temperature, max_mean_temp, holiday, attendin]])

        # Assuming additional features need to be filled
        #additional_features = np.zeros((1, 12))  # Adjust this based on actual feature engineering
        #input_data = np.hstack([input_data, additional_features])

        # Standardize the input data
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted number of attendees: {int(prediction[0][0])}")

# Call the function to make predictions
neuralNetworksPredictions()