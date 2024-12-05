import pandas as pd
import numpy as np
import pickle
import json
import os

# THIS FILE IS NOT BEING USED IN THE FINAL MODEL
# DONT RUN THIS FILE

# Load the pickle data
with open('../../Data/HouseData/cleanedMergedDf.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']
le_sex = data['le_sex']
le_membership_name = data['le_membership_name']


# Cache weather data to avoid multiple file reads
weather_cache = {}


def loadWeatherData(date, municipalityId):
    dateformatted = date.strftime("%Y-%m-%d")
    municipalityId = int(municipalityId)
    formattedMunicipalityId = str(municipalityId).zfill(4)  # Add leading zero to the municipalityId

    # Return cached weather data if available
    cache_key = (dateformatted, formattedMunicipalityId)
    if cache_key in weather_cache:
        return weather_cache[cache_key]

    # Weather object
    weather = {
        "date": date,
        "municipalityId": municipalityId,
        "max_temp_w_date": None,
        "rain": None
    }

    # Open the file corresponding to the given date
    file_path = f"../../bulkWeatherData/{date.year}/{dateformatted}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                try:
                    # Parse the JSON data from the current line (line-by-line)
                    data = json.loads(line)
                    # Check if 'municipalityId' matches the provided value
                    if data["properties"].get("municipalityId") == formattedMunicipalityId:
                        # Append the important properties to the weather data
                        if data["properties"].get("parameterId") == "max_temp_w_date":
                            weather["max_temp_w_date"] = data["properties"].get("value")
                        elif data["properties"].get("parameterId") == "acc_precip":
                            weather["rain"] = data["properties"].get("value")
                except json.JSONDecodeError as e:
                    # Handle malformed JSON lines
                    print(f"Error decoding JSON: {e} in line: {line}")
    
    # Cache the result
    weather_cache[cache_key] = weather
    print(f"Loaded weather data for {dateformatted} - {formattedMunicipalityId}")
    print(weather)
    return weather


# Apply function to load the weather data for all rows
def weatherLoader(df):
    # Define a function to process each row using apply
    def get_weather_data(row):
        weather_object = loadWeatherData(row["timestamp"], row["Municipality Code"])
        return weather_object["max_temp_w_date"], weather_object["rain"]

    # Use apply to vectorize the operation
    df[["max_temp_w_date", "rain"]] = df.apply(get_weather_data, axis=1, result_type="expand")
    #df["max_temp_w_date"] = df.apply(get_weather_data, axis=1)
    return df


# Process the entire DataFrame
newDF = weatherLoader(mergedDf)

# Save the updated DataFrame with the weather data
with open('../../Data/HouseData/HouseDataWeather.pkl', 'wb') as file:
    data = {
        'dataframe': newDF,
        'le_sex': le_sex,
        'le_membership_name': le_membership_name
    }
    pickle.dump(data, file)
