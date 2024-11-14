import pickle
import json
import pandas as pd



with open('../Data/EventData.pkl', 'rb') as file:
    data = pickle.load(file)

# Lets collect the json data from the weatherData.json file
with open("../API/weatherData.json", "r") as file:
    weatherdata = json.load(file)
    
mergedDf = data['dataframe']
weatherdata = weatherdata



from datetime import datetime, timedelta

# Add a new column for temperature if it doesn't exist yet
mergedDf["Temperature"] = None  # Initialize the column with None (or NaN)

for i in range(len(mergedDf)):
    # API Date is always 1 day behind
    date = weatherdata[i]["features"][0]["properties"]["from"]
    print(f"Raw API Date: {date}")  # Debugging step

    # Split the date part and remove the time part
    date = date.split("T")[0]  
    date = date + " 00:00:00"  

    # Convert API date string to datetime
    api_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    # Subtract one day to adjust the API date
    api_date_adjusted = api_date - timedelta(days=1)
    
    # Convert adjusted API date to string format (YYYY-MM-DD)
    api_date_adjusted_str = api_date_adjusted.strftime("%Y-%m-%d")

    # Convert REAL DATE to string format (YYYY-MM-DD)
    event_date = mergedDf.iloc[i]["Event Date"].strftime("%Y-%m-%d")

    print("REAL DATE:", event_date)
    print("API DATE (adjusted):", api_date_adjusted_str)

    # Now compare the dates (without the time part)
    if api_date_adjusted_str == event_date:
        print("Match found!")
        
        # Extract temperature from the API response
        temp = weatherdata[i]["features"][0]["properties"]["value"]
        print(f"Temperature for {event_date}: {temp}")
        temp = float(temp)

        # Update the temperature in the DataFrame at the corresponding index
        mergedDf.at[i, "Temperature"] = temp
    else:
        print(f"Mismatch: REAL DATE ({event_date}) != API DATE ({api_date_adjusted_str})")
    
    # Extract municipality from the API response and check if it matches
    municipality = weatherdata[i]["features"][0]["properties"]["municipalityId"]
    municipality = municipality[1:]  # Remove the first character (leading zero)
    municipality = int(municipality)

    if municipality in mergedDf["Municipality Code"].values and api_date_adjusted_str == event_date:
        print(f"Updating Temperature for Municipality Code {municipality} on {event_date}")
    else:
        print("No match for municipality or date")

# Print the updated DataFrame with temperature values
print(mergedDf.head())

# Save the updated DataFrame to a new pickle file
mergedDf = mergedDf.dropna()
data['dataframe'] = mergedDf
with open('../Data/EventData_Temperature.pkl', 'wb') as file:
    pickle.dump(data, file)

print("Updated DataFrame saved to 'EventData_Temperature.pkl'")

""" temp = weatherdata["features"][0]["properties"]["value"]
municipalityCode = weatherdata["features"][0]["properties"]["municipalityId"]


municipalityCode = municipalityCode[1:]
municipalityCodeInt = int(municipalityCode)




print(municipalityCodeInt)
print(temp)
print(mergedDf["Municipality Code"]) """