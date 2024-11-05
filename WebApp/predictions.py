import streamlit as st
import pandas as pd
import numpy as np
import pickle

def model_load():
    with open('../Data/model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = model_load()

modelLoaded = data['model']
le_EventTypeLoaded = data['le_EventType']
le_attendingWhatLoaded = data['le_attendingWhat']
le_GenderLoaded = data['le_Gender']
le_ZonesLoaded = data['le_Zones']


def showPredictions():
    st.title("Predictions using Random Forest Regressor")
    st.write("""###Test123""")
    
    eventTypes = {
        'Street Football',
        'GAME Girl Zone',
        'Street Basketball',
        'Street Dance',
        'Street Dance, Street Basketball, Street Football, GAME Girl Zone, Skate, Skateboarding',
        'Skate',
        'Street Basketball, Street Football'}
    
    attendingWhat = {
        'No',
        'Yes, Something else than sport',
        'Yes, Sport'}
    
    genders = {
        'Others',
        'Male',
        'Female'}

    zones = {
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
        'Stensbjergparken - Sønderborg'}
    
    dayOfweek = {
        0,1,2,3,4,5,6
    }
    
    eventType = st.selectbox("Event Type", eventTypes)
    zone = st.selectbox("Zone", zones)
    
    ok = st.button("Predict")
    if ok:
        eventType = le_EventTypeLoaded.transform([eventType])
        zone = le_ZonesLoaded.transform([zone])
        prediction = modelLoaded.predict([[eventType[0], zone[0]]])
        st.write(f"Predicted number of participants: {prediction[0]}")