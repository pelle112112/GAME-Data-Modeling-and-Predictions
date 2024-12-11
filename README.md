# GAME-Data-Modeling-and-Predictions

## Setup

### Following libraries are being used and needs to be installed

pandas, numpy, pickle, json, matplotlib, seaborn, sklearn, streamlit, keras, tensorflow, st-pages

### First set up a virtual environment for the libraries

### Next, install the libraries by running the following commands

pip install pandas

pip install numpy

pip install pickle

pip install matplotlib

pip install seaborn

pip install scikit-learn

pip install streamlit

pip install keras

pip install tensorflow

pip install st-pages

### If you want to use the prediction with weatherdata, then you need to set up the API key

Navigate to the following page and get the key (Its free) "https://opendatadocs.dmi.govcloud.dk/DMIOpenData"
Add it into a file in this location "/API/weatherAPIkey.txt"

## To run the app

Navigate to "/Webapp"

Type into the terminal: "streamlit run app.py"

## To update repo with large pkl files

Type into terminal: ❯ git lfs migrate import --include="\*.pkl"
