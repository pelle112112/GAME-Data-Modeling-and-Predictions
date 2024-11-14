import pickle   
import pandas as pd

# Load data
with open('../Data/EventData_Temperature_MaxMean.pkl', 'rb') as file:
    data = pickle.load(file)

# Read holidays data
holidayDF = pd.read_csv("../Data/publicholiday.DK.2024.csv")
holidayDF["Date"] = pd.to_datetime(holidayDF["Date"])

# Initialize merged DataFrame
mergedDf = data['dataframe']
mergedDf = mergedDf.dropna()

# Reset index to ensure it's sequential
mergedDf = mergedDf.reset_index(drop=True)

# Print data types for checking
print(holidayDF.dtypes)

# Initialize the Holiday column with 0
mergedDf["Holiday"] = 0

# Efficient approach for matching holidays
for i in range(len(mergedDf)):
    event_date = mergedDf.loc[i, "Event Date"]  # Using loc with the reset index
    # Check if the event date is in the holiday list
    if event_date in holidayDF["Date"].values:
        mergedDf.loc[i, "Holiday"] = 1
        print("Holiday found", event_date)
    else:
        print("No holiday found", event_date)
        
data = {
    'dataframe': mergedDf,
    'holidayDF': holidayDF
}

# Save the updated DataFrame
with open('../Data/EventData_Holiday.pkl', 'wb') as file:
    pickle.dump(data, file)