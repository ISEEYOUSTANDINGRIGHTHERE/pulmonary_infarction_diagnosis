import numpy as np
import cv2

def normalize(volume):
    min_val = np.min(volume)
    max_val = np.max(volume)
    return ((volume - min_val) / (max_val - min_val)).astype("float32")

def resize_volume(volume, desired_depth=64, desired_height=128, desired_width=128):
    current_depth = volume.shape[0]

    # Resize depth
    if current_depth > desired_depth:
        start = (current_depth - desired_depth) // 2
        volume = volume[start:start + desired_depth]
    elif current_depth < desired_depth:
        pad = (desired_depth - current_depth) // 2
        volume = np.pad(volume, ((pad, desired_depth - current_depth - pad), (0, 0), (0, 0)), mode='constant')

    # Resize each slice to desired H x W
    volume_resized = np.stack([cv2.resize(slice, (desired_width, desired_height)) for slice in volume])

    return volume_resized

def get_views(volume, size=(128, 128), depth=64):
    axial = resize_volume(volume, desired_depth=depth, desired_height=size[0], desired_width=size[1])
    coronal = resize_volume(np.transpose(volume, (1, 0, 2)), desired_depth=depth, desired_height=size[0], desired_width=size[1])
    sagittal = resize_volume(np.transpose(volume, (2, 0, 1)), desired_depth=depth, desired_height=size[0], desired_width=size[1])
    return axial, coronal, sagittal
