import pickle   
import pandas as pd

# Load data
with open('../../Data/HouseData/HouseDataWeather.pkl', 'rb') as file:
    data = pickle.load(file)

# Read holidays data
holidayDF = pd.read_csv("../../Data/publicholiday.DK.2024.csv")
holidayDF["Date"] = pd.to_datetime(holidayDF["Date"]).dt.date  # Ensure the Date is just the date part

# Initialize merged DataFrame
mergedDf = data['dataframe']
le_sex = data['le_sex']
le_membership_name = data['le_membership_name']

# Reset index to ensure it's sequential
mergedDf = mergedDf.reset_index(drop=True)

# Initialize the Holiday column with 0
mergedDf["Holiday"] = 0

# Remove the time from the timestamp, but save the original timestamp
mergedDf["date"] = mergedDf["timestamp"].dt.date  # Extract date only

# Efficient approach for matching holidays using `isin()`
mergedDf["Holiday"] = mergedDf["date"].isin(holidayDF["Date"]).astype(int)

# Print out some of the data to check if the "Holiday" column is populated correctly
print(mergedDf[["date", "Holiday"]].head())

# Drop the date column
mergedDf = mergedDf.drop(columns=["date"])

# Save the updated DataFrame
data = {
    'dataframe': mergedDf,
    'holidayDF': holidayDF,
    'le_sex': le_sex,
    'le_membership_name': le_membership_name

}

with open('../../Data/HouseData/HouseDataHolidayAndWeather.pkl', 'wb') as file:
    pickle.dump(data, file)
