import numpy as np
import cv2

def resize_volume(img, desired_shape=(128, 128, 64)):
    """Resize 3D image to desired shape"""
    depth, height, width = desired_shape
    img = cv2.resize(img, (width, height))
    img = np.resize(img, desired_shape)
    return img.astype(np.float32)

def normalize(volume):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume
