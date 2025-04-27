import numpy as np
import cv2
import pydicom

def load_dicom_image(filepath: str) -> np.ndarray:
    """Loads a DICOM file and returns the pixel array normalized to 0-1."""
    dcm = pydicom.dcmread(filepath)
    img = dcm.pixel_array.astype(np.float32)

    # Normalize to [0, 1]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def resize_image(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Resizes the image to the target size (default 224x224)."""
    resized_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return resized_img

def preprocess_image(filepath: str, target_size: int = 224) -> np.ndarray:
    """Loads, normalizes, and resizes a DICOM file for the model."""
    img = load_dicom_image(filepath)
    img = resize_image(img, target_size)

    # If grayscale, expand dims to (1, H, W)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)

    return img
