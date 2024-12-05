import pandas as pd
import numpy as np
import pickle


with open('../../Data/HouseData/HouseDataHolidayAndWeather.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']

# Find out all the dates where there is weather data
# Remove time from the timestamp
mergedDf['date'] = mergedDf['timestamp'].dt.date

daysWithWeather = mergedDf[mergedDf['max_temp_w_date'].notnull()]['date'].unique()
# Find out which years are in the dataset
years = mergedDf['timestamp'].dt.year.unique()
# Find out which years are missing weather data
yearsMissingWeather = [year for year in years if year not in mergedDf[mergedDf['max_temp_w_date'].notnull()]['timestamp'].dt.year.unique()]

print(f"Years missing weather data: {yearsMissingWeather}")

print(f"Number of days with weather data: {len(daysWithWeather)}")
print(f"number of days in the dataset: {len(mergedDf['date'].unique())}")


# Find out how many days are missing weather data
daysMissingWeather = [day for day in mergedDf['date'].unique() if day not in daysWithWeather]

print(f"Number of days missing weather data: {len(daysMissingWeather)}")

# Lets add missing weather data by filling it with the previous day's weather data

# First, we need to sort the data by date
mergedDf = mergedDf.sort_values(by='timestamp')

# Then we need to fill the missing weather data
mergedDf['max_temp_w_date'] = mergedDf['max_temp_w_date'].fillna(method='ffill')
mergedDf['rain'] = mergedDf['rain'].fillna(method='ffill')

# Check if there are any missing values left
missingValues = mergedDf.isnull().sum()
print(f"Missing values in the dataset: {missingValues[missingValues > 0]}")

