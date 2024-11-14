import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split


with open('../Data/EventData_Holiday.pkl', 'rb') as file:
    data = pickle.load(file)
    

mergedDf = data['dataframe']

# After having done both regression and classification, its now time to try out clustering
# Features (X) and target (y) for clustering
X = mergedDf[["Day of the week", "Event Type", "Zone", "Month", "max_mean_temp", "Holiday"]]  # features
y = mergedDf['Player Id_attendees']  # target variable (number of attendees)

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)

# Lets try out a few clustering models
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def kMeans():
    kmeans = KMeans(n_clusters=3, random_state=7)
    kmeans.fit(X_train)
    
    # Make predictions
    y_pred_kmeans = kmeans.predict(X_test)
    
    # Evaluate
    print('KMeans Silhouette Score:', silhouette_score(X_test, y_pred_kmeans))
    return kmeans

def agglomerativeClustering():
    agg = AgglomerativeClustering(n_clusters=3)
    agg.fit(X_train)
    
    # Make predictions
    y_pred_agg = agg.fit_predict(X_test)
    
    # Evaluate
    print('Agglomerative Clustering Silhouette Score:', silhouette_score(X_test, y_pred_agg))
    return agg

def dbscan():
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_train)
    
    # Make predictions
    y_pred_dbscan = dbscan.fit_predict(X_test)
    
    # Evaluate
    print('DBSCAN Silhouette Score:', silhouette_score(X_test, y_pred_dbscan))
    return dbscan

kmeans = kMeans()
agg = agglomerativeClustering()
dbscan = dbscan()



# Lets visualize the data to determine the amount of clusters
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.figure(figsize=(10, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
plt.title('KMeans Clustering')
plt.show()
