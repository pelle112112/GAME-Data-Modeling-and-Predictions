import pandas as pd
import pickle
import json
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os


# Load the pickle data
with open('../../Data/HouseData/cleanedMergedDf.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']
mergedDf['date'] = mergedDf['timestamp'].dt.date
le_sex = data['le_sex']
le_membership_name = data['le_membership_name']


# Preprocess a single weather file
def preprocess_weather_file(file_path, is_single_station=False):
    weather_dict = defaultdict(dict)
    
    # Check if the file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        properties = data.get("properties", {})
                        
                        if is_single_station:
                            # Handling single station data (2021, 2022, 2023)
                            parameter_id = properties.get("parameterId")
                            value = properties.get("value")
                            
                            # Add the data to the dictionary, assuming only one station for those years
                            weather_dict["single_station"][parameter_id] = value
                        else:
                            # Handling multiple municipality data (2024 and beyond)
                            municipality_id = properties.get("municipalityId")
                            parameter_id = properties.get("parameterId")
                            value = properties.get("value")
                            
                            if municipality_id and parameter_id:
                                weather_dict[municipality_id][parameter_id] = value
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            print(f"File not found: {file_path}")
    else:
        print(f"Weather file missing: {file_path}")
    
    return weather_dict


# Function for multiprocessing
def process_file(args):
    date, base_path, is_single_station = args
    dateformatted = date.strftime("%Y-%m-%d")
    file_path = f"{base_path}/{date.year}/{dateformatted}.txt"
    
    # Log missing files
    if not os.path.exists(file_path):
        print(f"Weather data file missing for date: {dateformatted}")
    
    return (dateformatted, preprocess_weather_file(file_path, is_single_station))


# Preprocess all required weather data
def preprocess_all_weather_data(dates, base_path, is_single_station=False):
    weather_data = {}
    
    # Prepare arguments for multiprocessing
    args = [(date, base_path, is_single_station) for date in dates]
    
    with Pool(min(cpu_count(), len(dates))) as pool:
        results = pool.map(process_file, args)

    counter = 0
    for dateformatted, data in results:
        weather_data[dateformatted] = data
        print(f"Processed weather data for {dateformatted}")
        
        # Log missing files (this will capture dates with no weather data)
        if not data:
            print(f"Missing weather data for {dateformatted}")
        
        counter += 1

    return weather_data


# Apply weather data to the DataFrame
def apply_weather_data(df, weather_data, is_single_station=False):
    def get_weather(row):
        # Get the formatted date (YYYY-MM-DD) for the row
        dateformatted = row["date"].strftime("%Y-%m-%d")
        
        if is_single_station:
            # For single station data (2021, 2022, 2023), use the data from the single station
            weather = weather_data.get(dateformatted, {}).get("single_station", {})
        else:
            # For multiple municipalities (2024 and beyond), use the weather data for the respective municipality
            municipality_id = f"0{int(row['Municipality Code'])}"
            weather = weather_data.get(dateformatted, {}).get(municipality_id, {})

        # Log the missing weather data for debugging
        if not weather:
            print(f"Missing weather data for date: {dateformatted}, Municipality Code: {row['Municipality Code'] if not is_single_station else 'N/A'}")
        
        # Return weather values, using `float('nan')` if missing
        return {
            "mean_temp": weather.get("mean_temp", float('nan')),
            "mean_daily_max_temp": weather.get("mean_daily_max_temp", float('nan')),
            "rain": weather.get("acc_precip", float('nan'))  # Ensure correct parameter name
        }

    # Apply the weather data for each row in the dataframe
    weather_info = df.apply(get_weather, axis=1)
    
    # Assign the weather data to the appropriate columns
    df["max_mean_temp"] = weather_info.map(lambda x: x["mean_daily_max_temp"])
    df["rain"] = weather_info.map(lambda x: x["rain"])
    df["mean_temp"] = weather_info.map(lambda x: x["mean_temp"])
    
    return df



# Main Execution
if __name__ == "__main__":
    # Get unique dates from the merged dataframe
    unique_dates = mergedDf["date"].unique()  # Using unique() to get the unique dates in the dataframe

    # Determine whether we're dealing with single station data (2021-2023) or multiple municipalities (2024)
    current_year = datetime.now().year
    is_single_station = True if any(date.year < current_year for date in unique_dates) else False

    # Preprocess weather data for all unique dates
    base_path = "../../bulkWeatherData"
    weather_data = preprocess_all_weather_data(unique_dates, base_path, is_single_station)

    # Apply preprocessed weather data to the DataFrame
    newDF = apply_weather_data(mergedDf, weather_data, is_single_station)

    # Save the updated DataFrame
    with open('../../Data/HouseData/HouseDataWeather.pkl', 'wb') as file:
        data = {
            'dataframe': newDF,
            'le_sex': le_sex,
            'le_membership_name': le_membership_name
        }
        pickle.dump(data, file)

    # Test loading and print a sample
    with open('../../Data/HouseData/HouseDataWeather.pkl', 'rb') as file:
        data = pickle.load(file)

    print(data['dataframe'].head())
