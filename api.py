import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app=FastAPI(title="ProMaint Predictive Maintenance API",
            description="An API for Predictive Maintenance using Random Forest Model")
# Load the trained model
model = joblib.load("models/model_rf.joblib")

class MaintenanceRequest(BaseModel):
    Air_temperature:float
    Process_temperature:float
    Tool_wear:int
    TWF:int
    HDF:int
    PWF:int
    OSF:int
    RNF:int
    Torque_Flage:int
    Rotational_Speed_Falge:int
    Wear_Severity_Level:int
    Heat_Dissipation_Gap:float
    Power_Indicator:float
    Type:str

def encode_type(machine_type: str):
    if machine_type == "L":
        return [1, 0, 0]
    elif machine_type == "M":
        return [0, 1, 0]
    elif machine_type == "H":
        return [0, 0, 1]
    else:
        raise ValueError("Type must be L, M, or H")

@app.post("/predict")
def predict_maintenance(request: MaintenanceRequest):
    input_data = np.array([[request.Air_temperature,
                            request.Process_temperature,
                            request.Tool_wear,
                            request.TWF,
                            request.HDF,
                            request.PWF,
                            request.OSF,
                            request.RNF,
                            request.Torque_Flage,
                            request.Rotational_Speed_Falge,
                            request.Wear_Severity_Level,
                            request.Heat_Dissipation_Gap,
                            request.Power_Indicator,
                            *encode_type(request.Type)]])
    
    prediction = model.predict(input_data)
    label_map = {0: "Normal", 1: "Failure"}


    return {"prediction_code": int(prediction[0]),"prediction_label": label_map[int(prediction[0])]}

