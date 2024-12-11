import pickle   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


image1 = plt.imread('Data/HouseData/pics/fig4.png')
image2 = plt.imread('Data/HouseData/pics/fig5.png')
image3 = plt.imread('Data/HouseData/pics/fig6.png')
image4 = plt.imread('Data/HouseData/pics/fig7.png')
image5 = plt.imread('Data/HouseData/pics/fig8.png')
image6 = plt.imread('Data/HouseData/pics/fig1.png')
image7 = plt.imread('Data/HouseData/pics/fig2.png')
image8 = plt.imread('Data/HouseData/pics/fig3.png')

st.title("House Data Exploration")
st.write("""### Data Exploration""")
st.write("""The data is cleaned, so we can do some exploratory data analysis""")
st.write("We can use the following image to determine when the most attendees are present, and therefore found out when the GAME houses are most popular and need the most staff.")
st.image(image1, caption='Average number of attendees pr hour', width=1000 )
st.write("____________________________________________________________")
st.write("By grouping the data by weekdays and weekends, we can see that the difference of popular visiting hours is quite significant.")
st.image(image2, caption='Number of attendees per hour on Weekdays vs Weekends', width=1000)
st.write("____________________________________________________________")
st.write("This graph shows the same as before, but its now based on average values, which shows the same trend, but more clearly.")
st.image(image3, caption='Average number of attendees per hour on Weekdays vs Weekends', width=1000)
st.write("____________________________________________________________")
st.write("This graph doesnt show average, but the total number of attendees per hour for each GAME house. The houses mostly follow the same trend, but there are some small differences.")
st.image(image4, caption='Attendees per hour for each GAME house', width=1000)
st.write("____________________________________________________________")
st.write("This graph shows the distribution of the number of attendees in a boxplot.")
st.image(image5, caption='Distribution of number of Attendees', width=1000)
st.write("____________________________________________________________")
st.write("This graph shows the correlation matrix of the daily attendance per house vs features.")
st.write("The highest correlation is: max_mean_temp, mean_temp and sex.")
st.image(image6, caption='Correlation Matrix - Daily attendance per house vs features', width=1000)
st.write("____________________________________________________________")
st.write("This graph shows the correlation matrix of the weekly attendance per house vs features. The correlation is a bit different when grouping the data by week.")
st.write("The highest correlation is: max_mean_temp, mean_temp and Holiday.")
st.image(image7, caption='Correlation Matrix - Weekly attendance per house vs features', width=1000)
st.write("____________________________________________________________")
st.write("This graph shows the correlation matrix of the seasonal attendance per house vs features. The correlation is again a bit different when grouping the data by season.")
st.write("The highest correlation is: max_mean_temp, mean_temp and Holiday.")
st.image(image8, caption='Correlation Matrix - Seasonal attendance per house vs features', width=1000)
