import streamlit as st
import numpy as np
import requests

st.title('Predictive Maintenance Prediction Application\n predict your machine status and how to maintain it')

# input features
Air_temperature=st.number_input('Air Temperature',min_value=0.0) 
Process_temperature=st.number_input('Process Temperature',min_value=0.0)  
Tool_wear=st.number_input('Tool wear',min_value=0)
TWF=st.radio("Tool wear failure", ['Yes','No'])   
HDF=st.radio('Heating Dissipation Failure',['Yes','No'])
PWF=st.radio('Power Failure',['Yes','No'])  
OSF=st.radio('Over Strain Failure',['Yes','No'])
RNF=st.radio('Random Failure',['Yes','No'])  
Torque_Flage=st.number_input('Machine Torque',min_value=0)    
Rotational_Speed_Falge=st.number_input('Rotational Speed',min_value=0)  
Wear_Severity_Level=st.number_input('Wear Severity Level',min_value=0)
Heat_Dissipation_Gap=st.number_input('Heat Dissipation Gap',min_value=0.0)
Power_Indicator=st.number_input('Power Indicator',min_value=0.0)
Type=st.selectbox('Type',['L','M','H'])

# prepare input data
TWF_val = 1 if TWF == "Yes" else 0
HDF_val = 1 if HDF == "Yes" else 0
PWF_val = 1 if PWF == "Yes" else 0
OSF_val = 1 if OSF == "Yes" else 0
RNF_val = 1 if RNF == "Yes" else 0

#prediction
if st.button("Predict"):
    payload = {
        "Air_temperature": Air_temperature,
        "Process_temperature": Process_temperature,
        "Tool_wear": Tool_wear,
        "TWF": TWF_val,
        "HDF": HDF_val,
        "PWF": PWF_val,
        "OSF": OSF_val,
        "RNF": RNF_val,
        "Torque_Flage": Torque_Flage,
        "Rotational_Speed_Falge": Rotational_Speed_Falge,
        "Wear_Severity_Level": Wear_Severity_Level,
        "Heat_Dissipation_Gap": Heat_Dissipation_Gap,
        "Power_Indicator": Power_Indicator,
        "Type": Type
    }

    #connect to api endpoint
    url='https://promaint-application-production.up.railway.app'
    response = requests.post(url,json=payload)
    if response.status_code == 200:
        result = response.json()
        if result["prediction_label"] == "Failure":
            st.error("⚠️ Prediction: FAILURE detected")
        else:
            st.success("✅ Prediction: NORMAL operation")
    else:
        st.error("Error calling the API.")



