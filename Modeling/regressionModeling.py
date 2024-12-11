import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from dataCleaning import mergedDf


with open('../Data/model.pkl', 'rb') as file:
    data = pickle.load(file)


cleanedDF = data["mergedDf"]
