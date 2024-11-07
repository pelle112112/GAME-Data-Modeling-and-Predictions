import requests
import json
import pickle
import pandas as pd

def saveWeatherData(data):
    with open("weatherData.json", "w") as file:
        json.dump(data, file)
    return

# The following api call returns the mean temperature for a given location and date
# https://dmigw.govcloud.dk/v2/climateData/collections/stationValue/items?api-key=e49d54d0-bd38-4612-8227-c270463007b8&parameterId=mean_temp&datetime=2023-11-01T00:00:00Z/2023-11-01T23:59:59Z&bbox=8.0,55.0,12.0,57.0
def getWeatherData(location, date):
    # The api key is stored in a separate file
    with open("weatherAPIkey.txt", "r") as file:
        api_key = file.read()
    
    # The location is the municipality id, which is a 4 digit number, but we need to add a 0 in front of it, and we need to convert it to a string
    locationString = str(location)
    locationString = "0" + locationString
    dateString = pd.to_datetime(date).strftime('%Y-%m-%d')
    dateString = dateString + "T17:00:00Z/" + dateString + "T23:59:59Z"

    try:
        url = "https://dmigw.govcloud.dk/v2/climateData/collections/municipalityValue/items?api-key="+api_key+"&municipalityId="+locationString+"&parameterId=mean_temp&datetime="+dateString+"&timeResolution=day"
        response = requests.get(url)
    
    except requests.exceptions.RequestException as e:
        print("Error: ", e)
    
    # Insert the json responses into an array and save it to a file
    data = response.json()
    # Lets get the mean temperature for the event
    #saveWeatherData(data)
    return data


with open('../Data/EventData.pkl', 'rb') as file:
    data = pickle.load(file)
    
mergedDf = data['dataframe']

# Lets get the weather data for one of the events
eventToTest = mergedDf.iloc[0]
location = eventToTest["Municipality Code"]
date = eventToTest["Event Date"]

weatherData = getWeatherData(location, date)

print(weatherData)

# Lets get the mean temperature for the event
meanTemp = weatherData["features"][0]["properties"]["value"]
