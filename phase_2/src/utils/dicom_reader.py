import os
import pydicom
import numpy as np
import nibabel as nib
from typing import Tuple

def load_dicom_series(dicom_dir: str) -> Tuple[np.ndarray, list]:
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f))
              for f in os.listdir(dicom_dir) if f.endswith(".dcm")]

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    volume = np.stack([s.pixel_array for s in slices], axis=0)

    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    return volume, slices

def save_nifti(volume: np.ndarray, output_path: str):
    nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nifti_img, output_path)
