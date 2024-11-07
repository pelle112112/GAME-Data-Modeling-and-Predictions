from weatherAPI import getWeatherData
import time
import pickle
import json

# This script is used to get the weather data for all the events in the dataset
# There is a 400 request limit per 5 seconds, so we need to sleep for 5 seconds after every 400 requests

# Load the data
with open('../Data/EventData.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']

# Get the unique dates and locations
uniqueDates = mergedDf["Event Date"].unique()
uniqueLocations = mergedDf["Municipality Code"].unique()

# Get the weather data for the first 400 events to test the script
sleepCounter = 0
weatherData = []
for i in range(len(mergedDf)):
    eventToTest = mergedDf.iloc[i]
    location = eventToTest["Municipality Code"]
    date = eventToTest["Event Date"]

    weatherData.append(getWeatherData(location, date))
    sleepCounter += 1
    print("Event", i, "done")
    print(weatherData[i])
    if sleepCounter == 400:
        time.sleep(5)
        sleepCounter = 0

# Save the weather data
def saveWeatherData(data):
    with open("weatherData.json", "w") as file:
        json.dump(data, file)
    return

saveWeatherData(weatherData)
print("Weather data saved")