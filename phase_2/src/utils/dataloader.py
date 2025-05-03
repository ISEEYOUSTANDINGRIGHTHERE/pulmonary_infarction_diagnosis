import os
import torch
from torch.utils.data import Dataset
from src.utils.dicom_loader import load_dicom_volume, get_views

LABEL_MAP = {
    "NoPI": 0,
    "Stage1": 1,
    "Stage2": 2,
    "Stage3": 3
}

class DicomVolumeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label_name, label_id in LABEL_MAP.items():
            label_folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_folder):
                continue
            for patient in os.listdir(label_folder):
                patient_path = os.path.join(label_folder, patient)
                if os.path.isdir(patient_path):
                    self.samples.append((patient_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        volume = load_dicom_volume(path)
        axial, _, _ = get_views(volume)
        volume_tensor = torch.tensor(axial, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
        return volume_tensor, label
