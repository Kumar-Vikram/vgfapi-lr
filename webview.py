import streamlit as st
import pandas as pd
import requests
import json

st.title("Prediction App")
st.write("this is a test")

df = pd.read_csv(r'C:\Users\Kumar.Vikram\Documents\Inceptez\Assignment\Data\data.csv')

#CONSOLE  = st.text_input("CONSOLE")
CONSOLE = st.selectbox("CONSOLE", pd.unique(df['CONSOLE']))
YEAR = st.number_input("YEAR", step=1, value=df['YEAR'].min())
CATEGORY = st.selectbox("CATEGORY", pd.unique(df['CATEGORY']))
PUBLISHER = st.selectbox("PUBLISHER", pd.unique(df['PUBLISHER']))
RATING = st.selectbox("RATING", pd.unique(df['RATING']))
CRITICS_POINTS = st.number_input("CRITICS_POINTS", step=0.1)
USER_POINTS = st.number_input("USER_POINTS", step=0.1)

inputs = {
"CONSOLE" : CONSOLE,
"YEAR" : YEAR,
"CATEGORY" : CATEGORY,
"PUBLISHER" : PUBLISHER,
"RATING" :  RATING,
"CRITICS_POINTS" : CRITICS_POINTS,
"USER_POINTS" : USER_POINTS
}

if st.button('Predict'):
    res = requests.post(url = "http://127.0.0.1:8000/predict", data = json.dumps(inputs))

    st.json(res.text)

# command to run in terminal: streamlit run webview.py    