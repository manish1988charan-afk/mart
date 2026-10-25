import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('model.sav')

model = load_model()

st.title("Advertising Sales Prediction App")
st.write("This app predicts sales based on advertising budgets for TV, Radio, and Newspaper.")

# User inputs
tv_budget = st.number_input("TV Advertising Budget ($)", min_value=0.0, value=100.0, step=1.0)
radio_budget = st.number_input("Radio Advertising Budget ($)", min_value=0.0, value=20.0, step=1.0)
newspaper_budget = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, value=10.0, step=1.0)

# Prediction trigger
if st.button("Predict Sales"):
    # Prepare the input features as a DataFrame to match training columns and avoid warnings
    input_data = pd.DataFrame({
        'TV': [tv_budget],
        'Radio': [radio_budget],
        'Newspaper': [newspaper_budget]
    })
    
    # Generate prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Sales: {prediction:.2f} units")
