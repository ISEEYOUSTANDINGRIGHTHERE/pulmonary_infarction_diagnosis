from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch
from src.utils.image_preprocessing import load_dicom, preprocess_image
from src.model_training_3d import PulmonaryModel

app = FastAPI()
model = PulmonaryModel()
model.load_state_dict(torch.load("models/saved_models/best_model.pth"))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_array = load_dicom(file.file)
    image_array = preprocess_image(image_array)
    image_tensor = torch.tensor(image_array).unsqueeze(0)
    
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).item()
    return {"Prediction": "Probable PI" if prediction == 1 else "No PI"}
