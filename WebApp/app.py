import streamlit as st
from RFModelpredictions import showRFPredictions

page = st.sidebar.selectbox("Choose page", ("Random Forest Regressor", "linear Regression"))

if page == "Random Forest Regressor":
    showRFPredictions()
elif page == "linear Regression":
    st.write("Need to add stuff")

