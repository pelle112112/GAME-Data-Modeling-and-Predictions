import pandas as pd
import numpy as np
import pickle
import json

with open('../Data/EventData_Holiday.pkl', 'rb') as file:
    data = pickle.load(file)
    
mergedDf = data['dataframe']

mergedDf["Temperature"] = mergedDf["Temperature"].astype(float)

import json

# Load the weather data from the bulk files and find the rain data for the given date and municipalityId
# The rain data is stored in the 'acc_precip_past24h' parameterId, which is the accumulated precipitation for the past 24 hours

def loadWeatherData(date, municipalityId):
    
    dateformatted = date.strftime("%Y-%m-%d")  
    print("Inputs: ", date, municipalityId)
    municipalityId = int(municipalityId)  
    formattedMunicipalityId = str(municipalityId)  
    formattedMunicipalityId = "0" + formattedMunicipalityId  # Add a leading zero to the municipalityId
    
    # Weather object
    weather = {
        "date": date,
        "municipalityId": municipalityId,
        "rain": None
    }

    
    # Open the file corresponding to the given date
    with open(f"../bulkWeatherData/{date.year}/{dateformatted}.txt", "r") as file:
        

        
        for line in file:
            try:
                # Parse the JSON data from the current line (line-by-line)
                data = json.loads(line)  # Convert the line (string) into a dictionary
                
                # Check if 'municipalityId' matches the provided value (from the 'properties' section)
                if data["properties"].get("municipalityId") == formattedMunicipalityId:
                    # Lets append the important properties to the weatherdata list.

                    if(data["properties"].get("parameterId") == "acc_precip"):
                        rain = data["properties"].get("value")
                        print("Rain: ", rain)

                        weather["rain"] = rain
                        print("Weather: ", weather)
                        
                        
            except json.JSONDecodeError as e:
                # Handle any malformed JSON lines by printing an error message
                print(f"Error decoding JSON: {e} in line: {line}")
                
    return weather

# Example usage:
weatherObject = loadWeatherData(mergedDf.iloc[0]["Event Date"], mergedDf.iloc[0]["Municipality Code"])
print(mergedDf.iloc[0]["Event Date"])
print(mergedDf.iloc[0]["Municipality Code"])


def mean_daily_max_tempLoader(df):
    counter = 0
    for i in range(len(df)):
        counter += 1
        print("Counter: ", counter)
        weatherObject = loadWeatherData(df.iloc[i]["Event Date"], df.iloc[i]["Municipality Code"])
        df.at[i, "rain"] = weatherObject["rain"]
    return df

newDF = mean_daily_max_tempLoader(mergedDf)

with open('../Data/rain.pkl', 'wb') as file:
    data = {
        'dataframe': newDF
    }
    pickle.dump(data, file)
    
with open('../Data/rain.pkl', 'rb') as file:
    data = pickle.load(file)
    

print(data['dataframe'])