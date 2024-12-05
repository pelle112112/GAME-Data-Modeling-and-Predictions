import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load in the pickle data
with open('../../Data/HouseData/HouseDataWeather.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']
le_sex = data['le_sex']
le_membership_name = data['le_membership_name']

print(mergedDf.dtypes)