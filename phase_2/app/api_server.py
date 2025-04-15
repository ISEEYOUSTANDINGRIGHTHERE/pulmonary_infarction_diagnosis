# app/api_server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import shutil, os
import torch
from src.utils.dicom_loader import load_dicom_series
from src.utils.image_preprocessing import resize_volume, normalize
from src.xai.grad_cam import GradCAM
from torch import nn
import numpy as np

app = FastAPI()

# Load model
model = torch.load("models/ct_model.pth", map_location=torch.device('cpu'))
target_layer = model.features[-1]  # adjust depending on your model
gradcam = GradCAM(model, target_layer)

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("app/templates/upload.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    save_path = f"temp_upload.dcm"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    volume, _ = load_dicom_series("data/sample_ct_scan/")  # replace if folder is uploaded
    volume = normalize(resize_volume(volume))
    volume_tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

    output = model(volume_tensor)
    prediction = torch.argmax(output).item()
    stage = "Early" if prediction == 0 else "Advanced"  # dummy logic

    cam = gradcam.generate_cam(volume_tensor)
    # Save a cam slice as preview (could be the middle slice)
    plt.imsave("app/static/cam_preview.png", cam[cam.shape[0]//2], cmap='jet')

    return {
        "prediction": "Probable Pulmonary Infarction" if prediction else "No PI",
        "stage": stage
    }
end stage.

qkash