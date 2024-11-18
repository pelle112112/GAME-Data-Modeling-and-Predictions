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

# Load player data
playerData = pd.read_excel('../Data/players2024-11-18 08-34-42.xlsx')

with open('../Data/EventData_Holiday.pkl', 'rb') as file:
    data = pickle.load(file)

eventDF = data['dataframe']

# Clean player data
playerData.drop(
    [
        'User Name', 'First Name', 'Last Name', 'Qr code ', 
        'Blood Type', 'Nationality', 'Has Health Problems', 
        'Has Disability', 'Disability  Type', 'Emergency  Contact Person Name',
        'Emergency  Phone Number', 'Allow Taking Photos', 'Player Phone Number'
    ], axis=1, inplace=True
)

# Change the types of the columns
playerData['Player Id'] = playerData['Player Id'].astype('float64')
playerData['Age'] = playerData['Age'].astype('float64')

# Merge the dataframes
mergedDf = pd.merge(eventDF, playerData, on='Player Id', how='inner')
mergedDf = mergedDf.drop('Problem Type', axis=1)
mergedDf = mergedDf.drop('Notes', axis=1)
mergedDf = mergedDf.dropna()

# Label encode 'Zone Id' as a categorical variable
labelencoder = LabelEncoder()
mergedDf['Zone Id'] = labelencoder.fit_transform(mergedDf['Zone Id'])

# Drop columns that won't be used for the regression model
mergedDf.drop(
    ['Event Id', 'Gender_y', 'Has Consent', 'Age_x', 'Gender_x', 'Age_y', 'Attending', 'Attending What_y', 'Attending What_x', 'Last Attendance', 'Country', 'Player Id', 'Event Date', 'Birthday Year', 'Zone Id'],
    axis=1, inplace=True
)

print(mergedDf.head())
print(mergedDf.info())

# Features (X) and target (y) for regression
# Select only numeric columns
X = mergedDf.select_dtypes(include=[np.number]).drop(['Player Id_attendees'], axis=1).to_numpy()  # Convert to NumPy array
y = mergedDf['Player Id_attendees'].to_numpy()  # Convert to NumPy array

# Check for NaN or infinite values in numeric columns only
if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    print("Data contains NaN values, cleaning up...")
    X = np.nan_to_num(X)  # Replace NaNs with 0
    y = np.nan_to_num(y)  # Replace NaNs with 0

# Convert to float32 (TensorFlow performs better with float32)
X = X.astype(np.float32)
y = y.astype(np.float32)

# Feature Scaling: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize the features

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

def optimizedTensorFlowModel():
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    # First hidden layer with dropout and BatchNormalization
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))  # Dropout to prevent overfitting
    model.add(layers.BatchNormalization())  # Batch Normalization
    
    # Second hidden layer with dropout
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))  # Dropout to prevent overfitting

    # Third hidden layer with dropout
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))  # Dropout to prevent overfitting

    # Output layer (for regression, no activation)
    model.add(layers.Dense(1))

    # Compile the model
    optimizer = optimizers.Adam(learning_rate=0.001)  # Adam optimizer with a learning rate
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    # Callbacks for early stopping and learning rate reduction
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), 
              callbacks=[early_stopping, reduce_lr], verbose=1)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    return model

# Train and evaluate the optimized regression model
model = optimizedTensorFlowModel()


# Save the model and label encoders)

data = {
    'scaler': scaler,
    'labelencoder': labelencoder,
    'dataframe': mergedDf
}

with open('../Data/Neural.pkl', 'wb') as file:
    pickle.dump(data, file)
    
model.save('../Data/NeuralModel.h5')
