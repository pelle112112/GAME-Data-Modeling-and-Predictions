import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split


with open('../Data/rain2.pkl', 'rb') as file:
    data = pickle.load(file)
    
mergedDf = data['dataframe']

# After having done regression, its now time to try out classificiation
# Features (X) and target (y) for classification
X = mergedDf[["Day of the week", "Event Type", "Zone", "Month", "max_mean_temp", "Holiday", "rain"]]  # features
y = mergedDf['Player Id_attendees']  # target variable (number of attendees)

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)

# Lets try out a few classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def randomForrestClassifier():
    rf_model = RandomForestClassifier(n_estimators=100, random_state=7, max_depth=20)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)

    # Evaluate
    print('Random Forest Classifier Accuracy:', accuracy_score(y_test, y_pred_rf))
    return rf_model

def decisionTreeClassifier():
    dt_model = DecisionTreeClassifier(random_state=7)
    dt_model.fit(X_train, y_train)

    # Make predictions
    y_pred_dt = dt_model.predict(X_test)

    # Evaluate
    print('Decision Tree Classifier Accuracy:', accuracy_score(y_test, y_pred_dt))
    return dt_model

def supportVectorMachine():
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred_svm = svm_model.predict(X_test)

    # Evaluate
    print('Support Vector Machine Classifier Accuracy:', accuracy_score(y_test, y_pred_svm))
    return svm_model

rf_model = randomForrestClassifier()
dt_model = decisionTreeClassifier()
svm_model = supportVectorMachine()

# With accuracy scores of 0.56, 0.55 and 0.11 respectively, the Random Forest Classifier is the best model, but still not very good
# This could be due to the fact that the number of attendees is a continuous variable, and not a categorical variable
# We could try to target a categorical variable instead, such as "High attendance" or "Low attendance"

# Lets create a new column called "Attendance" which will be a number from 0-1, where 0 is low attendance and 1 is high attendance
mergedDf["Attendance"] = np.where(mergedDf["Player Id_attendees"] > 14, 1, 0)

# Features (X) and target (y) for classification
X = mergedDf[["Day of the week", "Event Type", "Zone", "Month", "max_mean_temp", "Holiday"]]  # features
y = mergedDf['Attendance']  # target variable (number of attendees)

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)

rf_model = randomForrestClassifier()
dt_model = decisionTreeClassifier()
svm_model = supportVectorMachine()

# With accuracy scores of 0.875, 0.859 and 0.705, the models are now much better at predicting high or low attendance

# Lets save the best model and the data
data = {
    'dataframe': mergedDf,
    'rf_model': rf_model
}

with open('../Data/EventData_Attendance_Classification.pkl', 'wb') as file:
    pickle.dump(data, file)
    
