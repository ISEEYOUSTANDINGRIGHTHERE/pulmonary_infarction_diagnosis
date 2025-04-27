import torch
import numpy as np
import os

from models.cnn_model import PulmonaryInfarction3DCNN
from src.utils.image_preprocessing import load_dicom_image
from src.xai.grad_cam import generate_grad_cam  # Assuming this function exists!

# Load the model
def load_model(model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model = PulmonaryInfarction3DCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict on a DICOM folder
def predict_single_scan(dicom_folder: str, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # Preprocess the DICOM series into a 3D tensor
    scan_array = load_dicom_image(dicom_folder)  # Should return (D, H, W) np array
    scan_tensor = torch.tensor(scan_array).unsqueeze(0).unsqueeze(0).float()  # (1, 1, D, H, W)
    scan_tensor = scan_tensor.to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(scan_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities.squeeze().cpu().numpy()

# Full inference pipeline with Grad-CAM explanation
def full_inference_with_explanation(dicom_folder: str, model_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    
    predicted_class, class_probabilities = predict_single_scan(dicom_folder, model, device)
    
    # Generate Grad-CAM heatmap
    heatmap_path = generate_grad_cam(model, dicom_folder, predicted_class, device)
    
    return predicted_class, class_probabilities, heatmap_path
