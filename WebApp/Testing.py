import pickle
import pandas as pd

with open ('Data/rain.pkl', 'rb') as file:
    data = pickle.load(file)

rain = data['dataframe']

# Lets find the lowest and highest temperatures for each month
lowestTemp = rain.groupby('Month')['Temperature'].min()
highestTemp = rain.groupby('Month')['Temperature'].max()

print(rain.head())
print(rain.dtypes)
print(rain.isnull().sum())

print(lowestTemp)
print(highestTemp)

# Change none values to 0
rain["rain"] = rain["rain"].fillna(0)
print(rain.isnull().sum())
print(rain.head())
print(rain["rain"].value_counts())


# Save the model and label encoders
data = {
    'dataframe': rain}
with open('Data/rain2.pkl', 'wb') as file:
    pickle.dump(data, file)
