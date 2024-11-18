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


with open('../Data/EventData_Holiday.pkl', 'rb') as file:
    data1 = pickle.load(file)

holidayDF = data1


# Find out what the average amount of attendees are
print(mergedDf["Player Id_attendees"].mean())
# Median
print(mergedDf["Player Id_attendees"].median())
# What should be base a successful event on?
print(mergedDf["Player Id_attendees"].max())


with open('../Data/Neural.pkl', 'rb') as file:
    data2 = pickle.load(file)
    
neuralDF = data2['dataframe']

print(neuralDF.head())
print(neuralDF.columns)
print(neuralDF.dtypes)