import pandas as pd 
import numpy as np
import pickle
from keras import models, callbacks, optimizers
from keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras import callbacks

# Load the data

with open('../../Data/HouseData/regressionData.pkl', 'rb') as file:
    data = pickle.load(file)
    
mergedDf = data['mergedDf']
le_membership_name = data['le_membership_name']

mergedDf['month'] = mergedDf['timestamp'].dt.month
mergedDf['dayOfWeek'] = mergedDf['timestamp'].dt.dayofweek

X = mergedDf[['max_mean_temp', 'rain', 'mean_temp', 'Holiday', 'membership_name', 'month', 'dayOfWeek']]
y = mergedDf['Attendances']

scaler = StandardScaler()
X[['max_mean_temp', 'rain', 'mean_temp']] = scaler.fit_transform(X[['max_mean_temp', 'rain', 'mean_temp']])
le_membership_name = LabelEncoder()
X['membership_name'] = le_membership_name.fit_transform(X['membership_name'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


def tensorFlowModelImproved():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr])
    
    y_pred = model.predict(X_test).flatten()
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))
    
    return model, history

    
model, history = tensorFlowModelImproved()
import matplotlib.pyplot as plt
plt.hist(y, bins=50)
plt.show()


# Save the model and label encoders

data = {
    'model': model,
    'le_membership_name': le_membership_name,
    'history': history,
    'scaler': scaler,
    'dataframe': mergedDf
}

with open('../../Data/HouseData/Neural.pkl', 'wb') as file:
    pickle.dump(data, file)
    

