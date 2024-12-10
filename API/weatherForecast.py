import requests
import json
import pickle
import pandas as pd

def saveWeatherData(data):
    with open("weatherForecastData.json", "w") as file:
        json.dump(data, file)
    return

# Im defining the locations as coordinates based on their municipality code

coordinates = {
    101: [55.6761, 12.5683],
    851: [57.0488, 9.9217],
    561: [55.4760, 8.4647],
    791: [56.4538, 9.4026]
}
copenhagenCDSString = "POINT%2855.6761%2012.5683%29"
aalborgCDSString = "POINT%2857.0488%209.9217%29"
esbjergCDSString = "POINT%2855.4760%208.4647%29"
viborgCDSString = "POINT%2856.4538%209.4026%29"

CDSHashMap = {
    101: copenhagenCDSString,
    851: aalborgCDSString,
    561: esbjergCDSString,
    791: viborgCDSString
}


    

# The following api call returns the mean temperature, rain and mean max temperature for a given location and date
def getWeatherForecastData(date, hashmap, municipalityCode):
    # The api key is stored in a separate file
    with open("weatherForecastKey.txt", "r") as file:
        api_key = file.read()
        
    coordinatesString = str(coordinate[0]) + "%20" + str(coordinate[1])
    # Making a date that is 1 day after the event date
    date2 = pd.to_datetime(date) + pd.DateOffset(days=1)
    dateString = pd.to_datetime(date).strftime('%Y-%m-%dT%H:%M:%S')+ "Z" + "/" + date2.strftime('%Y-%m-%dT%H:%M:%S')+ "Z"
    print(dateString)
    try:
    
        newUrl = " https://dmigw.govcloud.dk/v1/forecastedr/collections/dkss_nsbs/cube?bbox=11,55,12,56&crs=crs84&parameter-name=water-temperature,salinity&datetime=2023-03-15T16:00:00.000Z/2023-03-15T19:00:00.000Z&api-key=<api-key>"
        url2 = " https://dmigw.govcloud.dk/v1/forecastedr/collections/dkss_nsbs/cube?bbox=6,26,12,68&crs=crs84&parameter-name=water-temperature,salinity,temperature-0m&datetime="+dateString+"&api-key="+api_key
        orgURL = "https://dmigw.govcloud.dk/v1/forecastedr/collections/harmonie_dini_sf/position?coords=POINT%2812.561%2055.715%29&parameter-name=temperature-0m&datetime=2023-03-15T16:00:00.000Z/2023-03-15T19:00:00.000Z&api-key=<api-key>"
        url = " https://dmigw.govcloud.dk/v1/forecastedr/collections/harmonie_dini_sf/position?coords="+hashmap[municipalityCode]+"&parameter-name=temperature-0m&datetime="+dateString+"&api-key="+api_key
        print(url2)
        response = requests.get(url2)
            
    except requests.exceptions.RequestException as e:
        print("Error: ", e)
        
    data = response.json()
    
    return data


with open('../Data/HouseData/regressionData.pkl', 'rb') as file:
    data = pickle.load(file)
    
mergedDf = data['mergedDf']

# Lets get the weather data for a future event next week
dayToTest = "2024-10-16"
coordinate = coordinates[101]
# Testing a day in next week
date = pd.to_datetime(dayToTest)
print(date)
print(CDSHashMap[561])

weatherData = getWeatherForecastData(date, CDSHashMap, 101)

print(weatherData)