import pandas as pd
import numpy as np
import pickle


# Load member data
memberData = pd.read_csv('../../Data/HouseData/fullMemberData.csv')


# Select the columns that will be used 
memberDf = memberData[['id', 'countrycode', 'status', 'usergroup_id', 'usergrouptype_id', 'membership_instances_id', 'num_admittances', 'latest_admittance', 'sex', 'birthdate', 'membership_name', 'active_from', 'active_to']]

# Drop rows with NaN values
memberDf = memberDf.dropna()

# Load admitted data
admittedData = pd.read_csv('../../Data/HouseData/20232024Admittance.csv')

# Select the columns that will be used 
admittanceDf = admittedData.drop(columns=["reason", "source"])



# Merge the member data into the admitted data
mergedDf = pd.merge(admittanceDf, memberDf, left_on='user_id', right_on='id', how='inner')

# Drop columns that won't be used 
mergedDf = mergedDf.drop(columns=['id_x', 'usergroup_id_x'])

# Rename columns
mergedDf = mergedDf.rename(columns={'id_y': 'id', 'usergroup_id_y': 'usergroup_id'})

# Remove all rows where the action is not 'granted'
mergedDf = mergedDf[mergedDf["action"] == "granted"]

# Drop columns that won't be used
# Action is just 'granted' for all rows
# Granted is just '1' for all rows
# Status is just 'active' for all rows
# countrycode is just 45 for all rows
mergedDf = mergedDf.drop(columns=['action', 'granted', 'status', 'usergrouptype_id', 'countrycode'])

# Convert types
mergedDf['timestamp'] = pd.to_datetime(mergedDf['timestamp'])
mergedDf['latest_admittance'] = pd.to_datetime(mergedDf['latest_admittance'])
mergedDf['birthdate'] = pd.to_datetime(mergedDf['birthdate'])
mergedDf['active_from'] = pd.to_datetime(mergedDf['active_from'])
mergedDf['active_to'] = pd.to_datetime(mergedDf['active_to'])


# Labelencode sex and membership_name

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
le_sex = LabelEncoder()
mergedDf["sex"] = le_sex.fit_transform(mergedDf["sex"])

le_membership_name = LabelEncoder()
#mergedDf["membership_name"] = le_membership_name.fit_transform(mergedDf["membership_name"])

# Save the model and label encoders
import pickle

data = {
    'dataframe': mergedDf,
    'le_sex':le_sex,
    'le_membership_name':le_membership_name
}

print(mergedDf.head())
print(mergedDf.info())
# Find the matching usergroup_id for each membership_name

# We know that the usergroup_id for GAME København is 4
# We know that the usergroup_id for GAME Aalborg is 38
# We know that the usergroup_id for GAME Esbjerg is 36
# We know that the usergroup_id for GAME Viborg is 37


# Add municipality code to the dataframe
counter = 0
# dictionary to map usergroup_id to Municipality Code
mapping = {
    4: 101,
    38: 851,
    36: 561,
    37: 791
}

# Map the Municipality Code directly using .map()
mergedDf["Municipality Code"] = mergedDf["usergroup_id"].map(mapping)
print(mergedDf["Municipality Code"].value_counts())
print(mergedDf.dtypes)


with open('../../Data/HouseData/cleanedMergedDf.pkl', 'wb') as file:
    pickle.dump(data, file)


