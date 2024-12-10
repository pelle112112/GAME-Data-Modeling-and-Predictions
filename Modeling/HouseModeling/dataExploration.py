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
# Lets see the distribution of the number of attendees
# First we need to group the data by each day in the different usergroup_id
# Then we can sum the number of attendees for each day
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns

def plotAttendeesPerDayAndHour():
    # Convert timestamp to date (if not already in datetime format)
    mergedDf['date'] = mergedDf['timestamp'].dt.date
    mergedDf['hour'] = mergedDf['timestamp'].dt.hour

    # Group by date and hour, then count the number of attendees
    grouped_by_hour = mergedDf.groupby(['date', 'hour']).size().reset_index(name='number_of_attendees')

    # Now, calculate the average participation per hour
    hourly_avg_attendance = grouped_by_hour.groupby('hour')['number_of_attendees'].mean().reset_index()

    # Plot the average number of attendees per hour
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=hourly_avg_attendance, x='hour', y='number_of_attendees', ax=ax)
    plt.title("Average Number of Attendees per Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Number of Attendees")
    plt.xticks(range(0, 24))  # Displaying all 24 hours
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()

    # Also, plot the daily number of attendees as before
    daily_attendees = grouped_by_hour.groupby('date')['number_of_attendees'].sum().reset_index()
    fig2 = plt.figure(figsize=(10, 6))
    sns.lineplot(data=daily_attendees, x='date', y='number_of_attendees')
    plt.title("Number of Attendees per Day")
    plt.xlabel("Date")
    plt.ylabel("Number of Attendees")
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()

    return fig, fig2

# Call the function to plot both the daily and hourly attendance


# Lets see the participation pr hour on weekdays vs weekends

def plotAttendeesPerHourWeekday():
    # Get the day of the week
    mergedDf['day_of_week'] = mergedDf['timestamp'].dt.dayofweek

    # Create a new column to indicate if the day is a weekend or not
    mergedDf['is_weekend'] = mergedDf['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Group by hour and day of the week
    grouped_by_hour_weekday = mergedDf.groupby(['hour', 'is_weekend']).size().reset_index(name='number_of_attendees')

    # Plot the number of attendees per hour on weekdays and weekends
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped_by_hour_weekday, x='hour', y='number_of_attendees', hue='is_weekend')
    plt.title("Number of Attendees per Hour on Weekdays vs Weekends")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Attendees")
    plt.xticks(range(0, 24))  # Displaying all 24 hours
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()

    return fig

# Call the function to plot the number of attendees per hour on weekdays vs weekends

# Lets do the same again, but with the average number of attendees
def averagePlotAttendeesPerHourWeekday():
    # Group by hour, day of the week, and date
    grouped_by_hour_weekday = mergedDf.groupby(['hour', 'is_weekend', 'date']).size().reset_index(name='number_of_attendees')

    # Calculate the average number of attendees per hour on weekdays and weekends
    avg_grouped_by_hour_weekday = grouped_by_hour_weekday.groupby(['hour', 'is_weekend'])['number_of_attendees'].mean().reset_index()

    # Plot the average number of attendees per hour on weekdays and weekends
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_grouped_by_hour_weekday, x='hour', y='number_of_attendees', hue='is_weekend')
    plt.title("Average Number of Attendees per Hour on Weekdays vs Weekends")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Number of Attendees")
    plt.xticks(range(0, 24))  # Displaying all 24 hours
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()

    return fig


# Now lets see the difference between the GAME houses (usergroup_id) pr hour
import matplotlib.pyplot as plt
import seaborn as sns

def plotAttendeesPerHouse():
    # Extract the hour from the timestamp if not done yet
    mergedDf['hour'] = mergedDf['timestamp'].dt.hour
    
    # Group by hour and usergroup_id (house), counting the number of attendees
    grouped_by_hour_house = mergedDf.groupby(['hour', 'usergroup_id']).size().reset_index(name='number_of_attendees')

    # Calculate the number of attendees per house
    avg_grouped_by_hour_house = grouped_by_hour_house.groupby(['hour', 'usergroup_id'])['number_of_attendees'].mean().reset_index()

    # Plot the number of attendees per hour for each house
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_grouped_by_hour_house, x='hour', y='number_of_attendees', hue='usergroup_id')
    plt.title("Average Number of Attendees per Hour for Each House")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Number of Attendees")
    plt.xticks(range(0, 24))  # Displaying all 24 hours
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()

    return fig



# Boxplot of the number of attendees per house pr day
def boxPlotAttendeesPerHouse():
    # Group by house and date, counting the number of attendees
    grouped_by_house_date = mergedDf.groupby(['usergroup_id', 'date']).size().reset_index(name='number_of_attendees')

    # Plot the number of attendees per house
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=grouped_by_house_date, x='usergroup_id', y='number_of_attendees')
    plt.title("Number of Attendees per House")
    plt.xlabel("House")
    plt.ylabel("Number of Attendees")
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()

    return fig


plotAttendeesPerDayAndHour()
plotAttendeesPerHourWeekday()
averagePlotAttendeesPerHourWeekday()
plotAttendeesPerHouse()
boxPlotAttendeesPerHouse()

# Lets calculate the correlation between the different columns
# Label encode membership_name
mergedDf['membership_name'] = le_membership_name.fit_transform(mergedDf['membership_name'])

# Calculate the correlation matrix
# We need to find out correlation between the amount of attendees and the other columns
# We first need to add the number of attendees to the dataframe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculateCorrelationHour():
    # Assuming 'mergedDf' is your DataFrame

    # Step 1: Extract the date from 'timestamp' and group by 'date' and 'usergroup_id'
    mergedDf['date'] = mergedDf['timestamp'].dt.date
    daily_attendance_per_house = mergedDf.groupby(['date', 'usergroup_id']).size().reset_index(name='num_attendees_per_day')

    # Step 2: Aggregate other features (e.g., weather, price, holiday) by 'date' and 'usergroup_id'
    aggregated_features = mergedDf.groupby(['date', 'usergroup_id']).agg({
        'max_mean_temp': 'mean',  # Average max temperature per house per day
        'rain': 'mean',  # Average rain per house per day
        'mean_temp': 'mean',  # Average mean temperature per house per day
        'Holiday': 'mean',  # Average holiday flag per house per day
        'price': 'mean',  # Average price per house per day
        'sex': 'mean'
    }).reset_index()

    # Step 3: Merge attendance data with other aggregated features
    data_for_correlation = daily_attendance_per_house.merge(aggregated_features, on=['date', 'usergroup_id'], how='left')

    # Step 4: Select only numeric columns for correlation analysis
    numeric_columns = data_for_correlation.select_dtypes(include=['number']).columns
    numeric_data = data_for_correlation[numeric_columns]

    # Step 5: Calculate correlation matrix for numeric columns
    correlation_matrix = numeric_data.corr()

    # Step 6: Plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix - Daily Attendance per House vs Features")
    plt.show()

    return correlation_matrix

# Run the correlation calculation and plot the heatmap
correlation_matrix = calculateCorrelationHour()

# Optionally, print out the correlation matrix
print(correlation_matrix)

def calculateCorrelationWeek():
    # Assuming 'mergedDf' is your DataFrame
    
    # Step 1: Extract the week and year from 'timestamp' to group by week
    mergedDf['year_week'] = mergedDf['timestamp'].dt.to_period('W')  # Create year-week period

    # Step 2: Group by 'year_week' and 'usergroup_id' to calculate weekly attendance
    weekly_attendance_per_house = mergedDf.groupby(['year_week', 'usergroup_id']).size().reset_index(name='num_attendees_per_week')

    # Step 3: Aggregate other features (e.g., weather, price, holiday) by 'year_week' and 'usergroup_id'
    aggregated_features = mergedDf.groupby(['year_week', 'usergroup_id']).agg({
        'max_mean_temp': 'mean',  # Average max temperature per house per week
        'rain': 'mean',  # Average rain per house per week
        'mean_temp': 'mean',  # Average mean temperature per house per week
        'Holiday': 'mean',  # Average holiday flag per house per week
        'price': 'mean',  # Average price per house per week
        'sex': 'mean'  # Average sex (or gender) ratio per house per week
    }).reset_index()

    # Step 4: Merge attendance data with other aggregated features
    data_for_correlation = weekly_attendance_per_house.merge(aggregated_features, on=['year_week', 'usergroup_id'], how='left')

    # Step 5: Select only numeric columns for correlation analysis
    numeric_columns = data_for_correlation.select_dtypes(include=['number']).columns
    numeric_data = data_for_correlation[numeric_columns]

    # Step 6: Calculate correlation matrix for numeric columns
    correlation_matrix = numeric_data.corr()

    # Step 7: Plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix - Weekly Attendance per House vs Features")
    plt.show()

    return correlation_matrix

# Run the correlation calculation and plot the heatmap
correlation_matrix = calculateCorrelationWeek()

# Optionally, print out the correlation matrix
print(correlation_matrix)


def calculateCorrelationSeason():
    # Assuming 'mergedDf' is your DataFrame

    # Step 1: Extract the date and season from 'timestamp'
    mergedDf['date'] = mergedDf['timestamp'].dt.date
    mergedDf['month'] = mergedDf['timestamp'].dt.month  # Extract the month for seasonal analysis

    # Step 2: Define seasons based on month
    # You can adjust the seasons as per your region, but here's a general way to categorize months into seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    mergedDf['season'] = mergedDf['month'].apply(get_season)

    # Step 3: Group by 'season' and 'usergroup_id' to get weekly attendees and average temperature
    weekly_attendance_per_house = mergedDf.groupby(['season', 'usergroup_id']).size().reset_index(name='num_attendees_per_season')
    
    # Aggregate other features (e.g., weather, price, holiday) by season and usergroup_id
    aggregated_features = mergedDf.groupby(['season', 'usergroup_id']).agg({
        'max_mean_temp': 'mean',  # Average max temperature per house per season
        'rain': 'mean',  # Average rain per house per season
        'mean_temp': 'mean',  # Average mean temperature per house per season
        'Holiday': 'mean',  # Average holiday flag per house per season
        'price': 'mean',  # Average price per house per season
        'sex': 'mean'
    }).reset_index()

    # Step 4: Merge attendance data with aggregated weather and other features
    data_for_correlation = weekly_attendance_per_house.merge(aggregated_features, on=['season', 'usergroup_id'], how='left')

    # Step 5: Select only numeric columns for correlation analysis
    numeric_columns = data_for_correlation.select_dtypes(include=['number']).columns
    numeric_data = data_for_correlation[numeric_columns]

    # Step 6: Calculate correlation matrix for numeric columns
    correlation_matrix = numeric_data.corr()

    # Step 7: Plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix - Seasonal Attendance per House vs Features")
    plt.show()

    return correlation_matrix

# Run the correlation calculation and plot the heatmap
correlation_matrix = calculateCorrelationSeason()

# Optionally, print out the correlation matrix
print(correlation_matrix)

def calculateDeltaAndAnalyzeEffectWithLag(mergedDf, month_temps):
    # Step 1: Extract month and year from the timestamp
    mergedDf['month'] = mergedDf['timestamp'].dt.month  # Extract month from timestamp
    mergedDf['year'] = mergedDf['timestamp'].dt.year  # Extract year if needed

    # Step 2: Map the average temperature for each month from the month_temps list
    avg_monthly_temps = {i + 2: month_temps[i][1] for i in range(len(month_temps))}  # Create a dictionary for average temperatures
    mergedDf['avg_month_temp'] = mergedDf['month'].map(avg_monthly_temps)

    # Step 3: Calculate the delta (difference between daily measured temp and average monthly temp)
    mergedDf['temp_delta'] = mergedDf['mean_temp'] - mergedDf['avg_month_temp']

    # Step 4: Create lagged features for temp_delta (1-day, 2-day, 3-day, and 7-day lags)
    mergedDf['temp_delta_lag_1'] = mergedDf['temp_delta'].shift(1)  # 1-day lag
    mergedDf['temp_delta_lag_2'] = mergedDf['temp_delta'].shift(2)  # 2-day lag
    mergedDf['temp_delta_lag_3'] = mergedDf['temp_delta'].shift(3)  # 3-day lag
    mergedDf['temp_delta_lag_7'] = mergedDf['temp_delta'].shift(7)  # 7-day lag

    # Step 5: Group data by date and usergroup_id to calculate attendance for each day
    daily_attendance = mergedDf.groupby(['date', 'usergroup_id']).size().reset_index(name='num_attendees_per_day')

    # Step 6: Merge the daily attendance with the temperature delta and lag features
    merged_with_delta = pd.merge(daily_attendance, mergedDf[['date', 'usergroup_id', 'temp_delta', 'temp_delta_lag_1', 'temp_delta_lag_2', 'temp_delta_lag_3', 'temp_delta_lag_7']], on=['date', 'usergroup_id'], how='left')

    # Step 7: Calculate correlation between lagged temp_delta and num_attendees_per_day
    correlation_matrix = merged_with_delta[['temp_delta', 'temp_delta_lag_1', 'temp_delta_lag_2', 'temp_delta_lag_3', 'temp_delta_lag_7', 'num_attendees_per_day']].corr()

    # Step 8: Visualize the correlation (optional)
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation between Temperature Delta (and Lags) and Attendance")
    plt.show()

    return correlation_matrix


# Example month_temps provided
month_temps = [
    (-31, 13, 2),  # January
    (-29, 16, 3),  # February
    (-27, 23, 5),  # March
    (-19, 29, 10),  # April
    (-8, 33, 16),  # May
    (-4, 36, 20),  # June
    (-1, 36, 20),  # July
    (-2, 35, 20),  # August
    (-6, 33, 20),  # September
    (-12, 27, 13),  # October
    (-22, 19, 8),  # November
    (-26, 15, 4)  # December
]

correlation_matrix = calculateDeltaAndAnalyzeEffectWithLag(mergedDf, month_temps)

# Optionally print the result
print(correlation_matrix)

# Save the dataframe to a pickle file


data = {
    'dataframe': mergedDf,
    'correlation_matrix': correlation_matrix
}
with open('../../Data/HouseData/finalMergedDf.pkl', 'wb') as file:
    pickle.dump(data, file)
    

    
