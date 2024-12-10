import pickle   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

with open('../Data/dataExplorationData.pkl', 'rb') as file:
    data = pickle.load(file)

le_EventTypeLoaded = data['le_EventType']
le_ZonesLoaded = data['le_Zones']
mergedDf = data['dataframe']

st.title("Data Exploration")
st.write("""### Data Exploration""")
st.write("""The data is cleaned, so we can do some exploratory data analysis""")

dataInfo = mergedDf.describe()
st.write(dataInfo)

datatypes = mergedDf.dtypes
st.write(datatypes)


mergedDf = mergedDf.drop(columns=['Event Id', 'Event Date', 'Player Id'])
st.write("""# Correlation matrix""")

correlationMatrix = mergedDf.corr()
# Display the correlation matrix with a heatmap
fig = plt.figure(figsize=(10, 6))
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm')
st.pyplot(fig)


st.write("""Lets see the distribution of the number of attendees""")
# Load the saved picture of the distribution of the number of attendees
image = plt.imread('../Data/pics/DistrubutionOfTheNumberOfAttendees.png')
st.image(image, caption='Distribution of the number of attendees', width=1000)

image2 = plt.imread('../Data/pics/pic2.png')
st.image(image2, caption='Number of attendees per event type', width=1000)


image3 = plt.imread('../Data/pics/pic3.png')
st.image(image3, caption='Number of attendees per zone', width=1000)