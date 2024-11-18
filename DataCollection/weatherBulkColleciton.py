import pandas as pd
import numpy as np
import pickle
import json

with open('../Data/EventData_Temperature.pkl', 'rb') as file:
    data = pickle.load(file)
    
mergedDf = data['dataframe']

mergedDf["Temperature"] = mergedDf["Temperature"].astype(float)

import json

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
        "mean_temp": None,
        "mean_daily_max_temp": None
    }
    
    # Open the file corresponding to the given date
    with open(f"../Data/bulkWeatherData/{date.year}/{dateformatted}.txt", "r") as file:
        for line in file:
            try:
                # Parse the JSON data from the current line (line-by-line)
                data = json.loads(line)  # Convert the line (string) into a dictionary
                
                # Check if 'municipalityId' matches the provided value (from the 'properties' section)
                if data["properties"].get("municipalityId") == formattedMunicipalityId:
                    # Lets append the important properties to the weatherdata list.
                    
                    if(data["properties"].get("parameterId") == "mean_daily_max_temp"):
                        temp = data["properties"].get("value")
                        weather["mean_daily_max_temp"] = temp
                        
                    elif(data["properties"].get("parameterId") == "mean_temp"):
                        temp = data["properties"].get("value")
                        
                        weather["mean_temp"] = temp
                        
            except json.JSONDecodeError as e:
                # Handle any malformed JSON lines by printing an error message
                print(f"Error decoding JSON: {e} in line: {line}")
                
    return weather

# Example usage:
weatherObject = loadWeatherData(mergedDf.iloc[0]["Event Date"], mergedDf.iloc[0]["Municipality Code"])
print(mergedDf.iloc[0]["Event Date"])
print(mergedDf.iloc[0]["Municipality Code"])


def mean_daily_max_tempLoader(df):
    for i in range(len(df)):
        weatherObject = loadWeatherData(df.iloc[i]["Event Date"], df.iloc[i]["Municipality Code"])
        df.at[i, "max_mean_temp"] = weatherObject["mean_daily_max_temp"]
    return df

newDF = mean_daily_max_tempLoader(mergedDf)

with open('../Data/EventData_Temperature_MaxMean.pkl', 'wb') as file:
    data = {
        'dataframe': newDF
    }
    pickle.dump(data, file)
    
with open('../Data/EventData_Temperature_MaxMean.pkl', 'rb') as file:
    data = pickle.load(file)
    
