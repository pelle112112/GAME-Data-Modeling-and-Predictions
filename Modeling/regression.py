import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score



with open('../Data/EventData_Holiday.pkl', 'rb') as file:
    data = pickle.load(file)
    

mergedDf = data['dataframe']
mergedDf = mergedDf.dropna()

# Features (X) and target (y) for regression
X = mergedDf[["Day of the week", "Event Type", "Zone", "Month", "max_mean_temp", "Holiday"]]  # features
y = mergedDf['Player Id_attendees']  # target variable (number of attendees)

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)

def linRegression():
    # Initialize and train the model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)


    # Evaluate the model
    print('Linear Regression Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Linear Regression R^2:', r2_score(y_test, y_pred))
    return regressor


def randomForrestRegressor():
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
    '''
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    
    print('Random Forest Mean Squared Error:', mean_squared_error(y_test, grid_search.predict(X_test)))
    print('Random Forest R^2:', r2_score(y_test, grid_search.predict(X_test)))
    '''
    return rf_model


def polynomialRegression ():
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_train)
    poly.fit(X_poly, y_train)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y_train)
    y_pred = lin2.predict(poly.fit_transform(X_test))
    print('Poly regression Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Poly regression R^2:', r2_score(y_test, y_pred))

    return lin2

def lassoRegression():
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    print('Lasso regression Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Lasso regression R^2:', r2_score(y_test, y_pred))
    return lasso

def decisionTreeRegressor():
    from sklearn.tree import DecisionTreeRegressor
    dt_model = DecisionTreeRegressor(random_state=7)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    print('Decision Tree Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Decision Tree R^2:', r2_score(y_test, y_pred))
    return dt_model


# lets test which features are the most important
def featureImportance():
    rf_model = randomForrestRegressor()
    feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                    index = X.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)
    
featureImportance()

linregressor = linRegression()
rf_model = randomForrestRegressor()
polyRegressor = polynomialRegression()
lassoRegressor = lassoRegression()
dtRegressor = decisionTreeRegressor()

modelToFile = {
    'rf_model' : rf_model,
    'linRegression' : linregressor}

with open('../Data/regressionModels.pkl', 'wb') as file:
    pickle.dump(modelToFile, file)