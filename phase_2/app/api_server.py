from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from src.utils.dicom_reader import load_dicom_series
from src.utils.dicom_loader import normalize, resize_volume
import os
import torch

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Dummy model placeholder
def dummy_model(volume_tensor):
    return torch.tensor([0.7])  # replace with your actual model later

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("result.html", {"request": request, "result": None})

@app.get("/predict_all", response_class=HTMLResponse)
def predict_all_patients(request: Request):
    parent_dir = "D:\\PulmonaryInfarction\\Phase2dataset\\PAT002"
    results = {}

    for patient_folder in os.listdir(parent_dir):
        patient_path = os.path.join(parent_dir, patient_folder)

        if os.path.isdir(patient_path):
            try:
                volume, _ = load_dicom_series(patient_path)
                volume = normalize(resize_volume(volume))
                volume_tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0)
                prediction = dummy_model(volume_tensor)
                results[patient_folder] = float(prediction.item())
            except Exception as e:
                results[patient_folder] = f"Error: {str(e)}"

    return templates.TemplateResponse("result.html", {"request": request, "result": results})
