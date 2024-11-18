import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras import models


def model_load():
    with open('../Data/Neural.pkl', 'rb') as file:
        data = pickle.load(file)
    with open('../Data/EventData.pkl', 'rb') as file2:
        labels = pickle.load(file2)
    return data, labels

data, labels = model_load()
        
le_EventTypeLoaded = labels['le_EventType']
le_attendingWhatLoaded = labels['le_attendingWhat']
le_GenderLoaded = labels['le_Gender']
le_ZonesLoaded = labels['le_Zones']

    
df = data['dataframe']
scaler = data['scaler']
labelencoder = data['labelencoder']
model = models.load_model('../Data/NeuralModel.h5')

def neuralNetworksPredictions():
    st.title("Predictions using Neural Networks")
    st.write("""###Test123""")
    
    eventTypes = (
        'Street Football',
        'GAME Girl Zone',
        'Street Basketball',
        'Street Dance',
        'Street Dance, Street Basketball, Street Football, GAME Girl Zone, Skate, Skateboarding',
        'Skate',
        'Street Basketball, Street Football')
    
    attendingWhats = (
        'No',
        'Yes, Something else than sport',
        'Yes, Sport')
    
    genders = (
        'Others',
        'Male',
        'Female')

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
    
    dayOfweeks = (
        'monday',
        'tuesday',
        'wednesday',
        'thursday',
        'friday',
        'saturday',
        'sunday'
    )
    def daySelection(day):
        if day == "monday":
            return 0
        elif day == "tuesday":
            return 1
        elif day == "wednesday":
            return 2
        elif day == "thursday":
            return 3
        elif day == "friday":
            return 4
        elif day == "saturday":
            return 5
        elif day == "sunday":
            return 6
        
    eventType = st.selectbox("Event Type", eventTypes)
    zone = st.selectbox("Zone", zones)
    #attendingWhatselection = st.selectbox("Attending What", attendingWhats)
    dayOfweekselection = st.selectbox("Day of the week", dayOfweeks)
    month = st.slider("Month", 1, 12, 1)
    temperature = st.slider("Temperature", -10, 40, 20)
    holiday = st.selectbox("Holiday", ["Yes", "No"])
    
    ok = st.button("Predict")
    if ok:
        zone = labelencoder["Zone"].transform([zone])
        eventType = labelencoder["Event Type"].transform([eventType])
        #attendingWhatselection = labelencoder["Attending What"].transform([attendingWhatselection])
        #gender = labelencoder
        dayOfweekselection = daySelection(dayOfweekselection)
        month = month
        temperature = temperature
        holiday = 1 if holiday == "Yes" else 0
        
        input_data = np.array([[zone, eventType, dayOfweekselection, month, temperature, holiday]])
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        st.write(prediction)
            
            
neuralNetworksPredictions()