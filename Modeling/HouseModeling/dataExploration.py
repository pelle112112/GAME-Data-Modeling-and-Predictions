import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load in the pickle data
with open('../../Data/HouseData/HouseDataHolidayAndWeather.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']
le_sex = data['le_sex']
le_membership_name = data['le_membership_name']


# Time to explore the data
print(mergedDf.head())
print(mergedDf.describe())
print(mergedDf.info())
print(mergedDf.columns)
print(mergedDf.dtypes)


print(mergedDf['mean_temp'].value_counts())
print(mergedDf['rain'].value_counts())