import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load in the pickle data
with open('../../Data/HouseData/HouseDataHolidayAndWeather.pkl', 'rb') as file:
    data = pickle.load(file)

mergedDf = data['dataframe']
mergedDf = mergedDf.drop("num_admittances", axis=1)
le_sex = data['le_sex']
le_membership_name = data['le_membership_name']


# Assuming mergedDf is your dataframe

# Step 1: Extract the year-week information
mergedDf['year_week'] = mergedDf['timestamp'].dt.to_period('W')  # Create a year-week period

# Step 2: Aggregate the number of attendees by week
weekly_attendance_per_house = mergedDf.groupby(['year_week', 'usergroup_id']).size().reset_index(name='num_attendees_per_week')

# Step 3: Aggregate other features for each week
aggregated_features_weekly = mergedDf.groupby(['year_week', 'usergroup_id']).agg({
    'max_mean_temp': 'mean',  # Average max temperature per week per house
    'rain': 'mean',  # Average rain per week per house
    'mean_temp': 'mean',  # Average mean temperature per week per house
    'Holiday': 'mean',  # Average holiday flag per week per house
    'price': 'mean',  # Average price per week per house
    'sex': 'mean'  # Average sex (or gender) ratio per house per week
}).reset_index()

# Merge the weekly attendance with other aggregated features
data_for_regression = weekly_attendance_per_house.merge(aggregated_features_weekly, on=['year_week', 'usergroup_id'], how='left')

# Step 4: Define the target variable and features
# Target variable: num_attendees_per_week
y = data_for_regression['num_attendees_per_week']

# Features: All columns excluding target and identifiers (year_week, usergroup_id)
X = data_for_regression.drop(['num_attendees_per_week', 'year_week', 'usergroup_id'], axis=1)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Instantiate and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')



import pandas as pd

# Assuming 'df' is your dataframe and 'num_attendees_per_day' holds daily attendance
# Group by 'usergroup_id' and 'date' to sum the attendance per house per day
mergedDf['date'] = pd.to_datetime(mergedDf['timestamp']).dt.date
mergedDf['Attendances'] = mergedDf.groupby(['usergroup_id', 'date'])['usergroup_id'].transform('count')
mergedDf['membership_name'] = le_membership_name.fit_transform(mergedDf['membership_name'])
X = mergedDf['']
y = mergedDf['Attendances']

from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(X, y, degree=2):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

polynomial_regression_model, mse, r2 = polynomial_regression(X, y, degree=2)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Polynomial regression model: {polynomial_regression_model}')