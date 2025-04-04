import pydicom
import numpy as np
import SimpleITK as stik

def load_dicom(path):
    dicom_image =pydicom.dcmread(path)
    return dicom_image.pixel_array    #converts dicom to numpy array

def load_nifti(path):
    nifti_iage=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(nifti_image) #convert nifti to numpy array- nifti is simpler format for analayi sand visualization sp we use this

def preprocess_image(image_array):
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))  # Normalize-image
    return np.expand_dims(image_array, axis=0)
