import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load in the data from pickle file



with open('../Data/rain2.pkl', 'rb') as file:
    data = pickle.load(file)

with open('../Data/model.pkl', 'rb') as file:
    data2 = pickle.load(file)


le_EventTypeLoaded = data2['le_EventType']
le_attendingWhatLoaded = data2['le_attendingWhat']
le_GenderLoaded = data2['le_Gender']
le_ZonesLoaded = data2['le_Zones']


mergedDf = data['dataframe']

# The data is cleaned, so we can do some exploratory data analysis
# Lets see the distribution of the number of attendees

plt.figure(figsize=(10, 6))
sns.histplot(mergedDf['Player Id_attendees'], bins=30, kde=True)
plt.title('Distribution of the number of attendees')
plt.xlabel('Number of attendees')
plt.ylabel('Frequency')
plt.show()

# Boxplot of the features
# Use of label encoders to get the original event type names for the plot
# We put it into a new dataframe so we dont change the original dataframe
df = mergedDf.copy()

# Make all float columns integers
df['Event Type'] = df['Event Type'].astype(int)
df['Zone'] = df['Zone'].astype(int)

df['Event Type'] = le_EventTypeLoaded.inverse_transform(df['Event Type'])
df['Zone'] = le_ZonesLoaded.inverse_transform(df['Zone'])


plt.figure(figsize=(10, 6))
sns.boxplot(x='Event Type', y='Player Id_attendees', data=df)
plt.title('Number of attendees per event type')
plt.xticks(rotation=45)
plt.show()


# Lets see the distribution of the number of attendees per zone
# We need to fix the x-axis labels so they are readable
plt.figure(figsize=(10, 6))
sns.boxplot(x='Zone', y='Player Id_attendees', data=df)
plt.title('Number of attendees per zone')
plt.xticks(rotation=90)
plt.show()



# Save the model, label encoders and the cleaned data
data = {
    'dataframe': mergedDf,
    'dataframeLE': df,
    'le_EventType': le_EventTypeLoaded,
    'le_attendingWhat': le_attendingWhatLoaded,
    'le_Gender': le_GenderLoaded,
    'le_Zones': le_ZonesLoaded}

with open('../Data/dataExplorationData.pkl', 'wb') as file:
    pickle.dump(data, file)

