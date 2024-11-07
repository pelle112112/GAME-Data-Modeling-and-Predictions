import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load in the csv files (contains 20000 rows of data)
df = pd.read_csv("../Data/players_attendance_report_5.11.2024_10.40.07.csv")
df2 = pd.read_csv("../Data/players_attendance_report_5.11.2024_10.40.46.csv")

# Removal of the non important columns
df = df[["Event Id", "Event Type", "Player Id", "Attending What", "Age", "Gender", "Event Date", "Zone"]]
df2 = df2[["Event Id", "Event Type", "Player Id", "Attending What", "Age", "Gender", "Event Date", "Zone"]]

# Merging of the two dataframes 
mergedDf = df.merge(df2, how="outer")

# Dropping rows where data is N/A (Was about 15 rows)
mergedDf = mergedDf.dropna()

# Checking out the number of participants in each zone
#print(mergedDf["Zone"].value_counts())

# Checking for the number of participants for each event type
#print(mergedDf["Event Type"].value_counts())


# Now lets merge all the zones together with less than 100 participants into "Other Zones"
def zoneShortener(Zones):
    zones_map = {}
    for i in range(len(Zones)):
        if Zones.values[i] >= 100:
            zones_map[Zones.index[i]] = Zones.index[i]
        else:
            zones_map[Zones.index[i]] = 'Other Zones'
        
    return zones_map

zones_map = zoneShortener(mergedDf.Zone.value_counts())
mergedDf["Zone"] = mergedDf["Zone"].map(zones_map)

# Now lets merge "Street Football, Gadefodbold with "Street Football", and remove parkour with its 1 participant

def eventTypeFixer(events):
    events_map = {}
    for i in range(len(events)):
        event_value = events.iloc[i]  # Access event value by position
        if event_value == "Street Football, Gadefodbold":
            events_map[event_value] = 'Street Football'
        elif event_value == "Parkour":
            events_map[event_value] = 'Street Dance, Street Basketball, Street Football, GAME Girl Zone, Skate, Skateboarding'
        else:
            events_map[event_value] = event_value  # Map event to itself if no conditions are met
    
    return events_map  # Make sure to return the map

# Apply the function to your Series
events_map = eventTypeFixer(mergedDf["Event Type"])

# Map the corrected event types back to your DataFrame
mergedDf["Event Type"] = mergedDf["Event Type"].map(events_map)

# Lets check the boxplots of the ages at each zone
"""fig,ax  = plt.subplots(1,1, figsize=(12,7))
mergedDf.boxplot('Age', 'Zone', ax=ax)
plt.ylabel('Age')
plt.xticks(rotation=90)
plt.show()
"""
# After reviewing the boxplot, we need to remove the outliers, but taking all ages above 40 and changing them to 40

def ageOutlierRemoval(x):
    if x >= 40:
        return 40
    return int(x)

mergedDf["Age"] = mergedDf["Age"].apply(ageOutlierRemoval)

fig,ax  = plt.subplots(1,1, figsize=(12,7))
mergedDf.boxplot('Age', 'Zone', ax=ax)
plt.ylabel('Age')
plt.xticks(rotation=90)
plt.show()
print(mergedDf["Event Type"].value_counts())

# Label Encoding
print(mergedDf["Zone"].unique())
from sklearn.preprocessing import LabelEncoder
le_EventType = LabelEncoder()
mergedDf["Event Type"] = le_EventType.fit_transform(mergedDf["Event Type"])

le_attendingWhat = LabelEncoder()
mergedDf["Attending What"] = le_attendingWhat.fit_transform(mergedDf["Attending What"])

le_Gender = LabelEncoder()
mergedDf["Gender"] = le_Gender.fit_transform(mergedDf["Gender"])

le_Zones = LabelEncoder()
mergedDf["Zone"] = le_Zones.fit_transform(mergedDf["Zone"])



# Lastly lets convert dates to dateformat
mergedDf["Event Date"] = pd.to_datetime(mergedDf["Event Date"])

print(mergedDf["Event Date"])
print(mergedDf.dtypes)


# Now its time to do some feature engineering

# Lets create a new column called "Day of the week" which will be a number from 0-6, where 0 is Monday and 6 is Sunday


mergedDf["Day of the week"] = mergedDf["Event Date"].dt.dayofweek

# Lets create a new column called "Month" which will be a number from 1-12, where 1 is January and 12 is December
mergedDf["Month"] = mergedDf["Event Date"].dt.month

# Lets check what day of the week has the most participants
print(mergedDf["Day of the week"].value_counts())
print(mergedDf["Month"].value_counts())

# The event id we have is not unique, so we need to make a new id with the eventid and the event date
newId = mergedDf["Event Id"].astype(str) + mergedDf["Event Date"].astype(str)

mergedDf["Event Id"] = newId

# Now we can find the number of participants for each event
mergedDf["Player Id_attendees"] = mergedDf.groupby("Event Id")["Player Id"].transform("count")


print(mergedDf["Player Id_attendees"].unique())

# Features (X) and target (y) for regression
X = mergedDf[["Attending What", "Day of the week", "Event Type", "Zone", "Month"]]  # features
y = mergedDf['Player Id_attendees']  # target variable (number of attendees)

# Splitting the data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)


# Initialize and train the model
from sklearn.linear_model import LinearRegression
# Initialize and train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)


# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2:', r2_score(y_test, y_pred))



from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=7, max_depth=20)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate
print('Random Forest Mean Squared Error:', mean_squared_error(y_test, y_pred_rf))
print('Random Forest R^2:', r2_score(y_test, y_pred_rf))

# Time to optimize the model
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
print('Random Forest Mean Squared Error:', mean_squared_error(y_test, grid_search.predict(X_test)))
print('Random Forest R^2:', r2_score(y_test, grid_search.predict(X_test)))


# Save the model and label encoders
import pickle

data = {
    'model': rf_model,
    'le_EventType': le_EventType,
    'le_attendingWhat': le_attendingWhat,
    'le_Gender': le_Gender,
    'le_Zones': le_Zones}
with open('../Data/model.pkl', 'wb') as file:
    pickle.dump(data, file)
    
with open('../Data/model.pkl', 'rb') as file:
    data = pickle.load(file)
    
modelLoaded = data['model']
le_EventTypeLoaded = data['le_EventType']
le_attendingWhatLoaded = data['le_attendingWhat']
le_GenderLoaded = data['le_Gender']
le_ZonesLoaded = data['le_Zones']

