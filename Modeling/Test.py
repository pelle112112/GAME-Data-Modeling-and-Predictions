import pickle   
import pandas as pd


with open('../Data/EventData_Temperature_MaxMean.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']

print(mergedDf.head())
print(mergedDf.count())
mergedDf = mergedDf.dropna()

# Find the oldest event
print(mergedDf["Event Date"].min())

print(mergedDf.isna().sum())