import streamlit as st
from RFModelpredictions import showRFPredictions
from ClassificationModels import showClassificationPredictions
from neuralNetworksModels import neuralNetworksPredictions

page = st.sidebar.selectbox("Predictions", ("Random Forest Regressor", "linear Regression", "Random Forest Classifier", "Neural Networks"))

if page == "Random Forest Regressor":
    showRFPredictions()
elif page == "linear Regression":
    st.write("Need to add stuff")

elif page == "Random Forest Classifier":
    showClassificationPredictions()

elif page == "Neural Networks":
    neuralNetworksPredictions()