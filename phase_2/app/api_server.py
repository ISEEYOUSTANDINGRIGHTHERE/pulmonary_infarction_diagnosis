from fastapi import FastAPI
from src.utils.dicom_reader import load_dicom_series
from src.utils.dicom_loader import normalize, resize_volume
import os
import torch

app = FastAPI()

# Dummy model — replace with your trained one
def dummy_model(volume_tensor):
    return torch.tensor([0.7])  # Placeholder

@app.get("/predict_all")
def predict_all_patients():
    parent_dir = "D:\\PulmonaryInfarction\\Phase2dataset"
    results = {}

    for patient_folder in os.listdir(parent_dir):
        patient_path = os.path.join(parent_dir, patient_folder)

        if os.path.isdir(patient_path):
            try:
                volume, _ = load_dicom_series(patient_path)
                volume = normalize(resize_volume(volume))  # resize & normalize
                volume_tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0)  # shape: [1,1,D,H,W]
                
                prediction = dummy_model(volume_tensor)
                results[patient_folder] = float(prediction.item())

            except Exception as e:
                results[patient_folder] = f"Error: {str(e)}"

<<<<<<< Updated upstream
    cam = gradcam.generate_cam(volume_tensor)
    # Save a cam slice as preview (could be the middle slice)
    plt.imsave("app/static/cam_preview.png", cam[cam.shape[0]//2], cmap='jet')

    return {
        "prediction": "Probable Pulmonary Infarction" if prediction else "No PI",
        "stage": stage
    }
end stage.

qkash
=======
    return results
>>>>>>> Stashed changes
