import pickle   


with open('../Data/EventData_Temperature.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']

print(mergedDf.head())