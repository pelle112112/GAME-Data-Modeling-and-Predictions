import streamlit as st
import pandas as pd
import numpy as np
import pickle


def model_load():
    with open('../Data/model.pkl', 'rb') as file:
        data = pickle.load(file)
    with open('../Data/EventData_Attendance_Classification.pkl', 'rb') as file2:
        classificationModels = pickle.load(file2)
    return data, classificationModels

data, classificationModels = model_load()


rfClassifier = classificationModels['rf_model']
le_EventTypeLoaded = data['le_EventType']
le_attendingWhatLoaded = data['le_attendingWhat']
le_GenderLoaded = data['le_Gender']
le_ZonesLoaded = data['le_Zones']

st.title("Event Attendance Prediction")

def showClassificationPredictions():
    st.write("### Random Forest Classifier")
    
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
        eventType = le_EventTypeLoaded.transform([eventType])
        zone = le_ZonesLoaded.transform([zone])
        #attendingWhatselection = le_attendingWhatLoaded.transform([attendingWhatselection])
        dayOfweekselection = daySelection(dayOfweekselection)
        month = month
        temperature = temperature
        holiday = 1 if holiday == "Yes" else 0
        prediction = rfClassifier.predict([[eventType[0], zone[0], dayOfweekselection, month, temperature, holiday]])
        if prediction[0] == 1:
            st.write("High Attendance (more than 14 attendees)")
        else:
            st.write("Low Attendance (less than 14 attendees)")