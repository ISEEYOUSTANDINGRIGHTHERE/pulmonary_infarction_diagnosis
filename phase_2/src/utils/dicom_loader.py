import os
import pydicom
import numpy as np
import cv2

def load_dicom_volume(dicom_folder_path):
    print(f"--- Loading DICOMs from: {dicom_folder_path} ---")
    slices = []
    dcm_count = 0
    file_count = 0

    # Check if path exists before listing
    if not os.path.isdir(dicom_folder_path):
         print(f"ERROR: Directory not found in load_dicom_volume: {dicom_folder_path}")
         raise FileNotFoundError(f"Directory not found: {dicom_folder_path}")

    for filename in os.listdir(dicom_folder_path):
        file_count += 1
        filepath = os.path.join(dicom_folder_path, filename)
        # Use case-insensitive check for .dcm
        if os.path.isfile(filepath) and filename.lower().endswith(".dcm"):
            # print(f"    -> Reading DICOM: {filename}") # Uncomment for verbose logging
            try:
                ds = pydicom.dcmread(filepath)
                # Basic check if it looks like a valid image slice
                if hasattr(ds, 'pixel_array') and hasattr(ds, 'InstanceNumber'):
                     slices.append(ds)
                     dcm_count += 1
                else:
                     print(f"    -> Skipped {filename}: Missing pixel_array or InstanceNumber tag.")
            except Exception as e:
                print(f"      Error reading {filepath}: {e}")
        # else:
        #      print(f"    -> Skipped non-file or non-dcm: {filename}") # Uncomment for verbose logging


    print(f"--- Found {dcm_count} valid DICOM files out of {file_count} total items. ---")

    if not slices:
        # Handle the case where no valid DICOM slices were found
        print(f"ERROR: No valid DICOM files found or loaded for {dicom_folder_path}.")
        raise ValueError(f"No valid DICOM files found in folder: {dicom_folder_path}")

    # Sort slices by InstanceNumber (ensure InstanceNumber exists)
    try:
        # Attempt to convert to int, handle potential non-integer values
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except AttributeError:
        print(f"Warning: Could not sort by InstanceNumber for {dicom_folder_path}. Some files might lack the tag.")
        # Decide on fallback behavior - e.g., sort by filename? Or proceed unsorted?
        # slices.sort(key=lambda x: x.filename) # Example: sort by filename if InstanceNumber fails
    except ValueError:
        print(f"Warning: Could not convert InstanceNumber to int for sorting in {dicom_folder_path}. Proceeding potentially unsorted.")
        # Proceed unsorted or sort by filename as fallback

    # Stack pixel arrays into 3D volume
    try:
        # Ensure all arrays have compatible shapes before stacking if necessary
        # This basic stack assumes all slices were read correctly and have same dimensions
        volume = np.stack([s.pixel_array for s in slices])
        print(f"Volume stacked, shape: {volume.shape}")
        # Apply normalization (ensure normalize function exists)
        volume = normalize(volume)
        return volume
    except ValueError as e:
        print(f"Error during np.stack for {dicom_folder_path}. Number of slices: {len(slices)}. Error: {e}")
        # Check shapes if needed: print([s.pixel_array.shape for s in slices])
        raise ValueError(f"Failed to stack DICOM slices - likely inconsistent dimensions. Error: {e}")
    except Exception as e:
        print(f"Unexpected error during stacking/normalization: {e}")
        raise e

def normalize(volume):
    # Ensure robust normalization, handle flat images (min==max)
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val > 1e-6: # Use a small epsilon
         volume = (volume - min_val) / (max_val - min_val)
    else:
         volume = np.zeros_like(volume) # Or set to 0.5, depending on desired behavior
    return volume.astype("float32")

def resize_volume(volume, desired_depth=64, desired_height=128, desired_width=128):
    # Ensure volume is 3D
    if volume.ndim != 3:
        raise ValueError(f"Input volume must be 3D, but got shape {volume.shape}")

    current_depth, current_height, current_width = volume.shape

    # Resize depth (center crop or pad)
    if current_depth > desired_depth:
        start = (current_depth - desired_depth) // 2
        volume = volume[start:start + desired_depth, :, :]
    elif current_depth < desired_depth:
        pad_before = (desired_depth - current_depth) // 2
        pad_after = desired_depth - current_depth - pad_before
        volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')

    # Resize H x W for each slice
    # Need to handle potential interpolation issues
    try:
        # Use INTER_LINEAR for general resizing, INTER_NEAREST for masks
        volume_resized = np.stack([
            cv2.resize(slice_img, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
            for slice_img in volume
        ])
    except Exception as e:
         print(f"Error during cv2.resize: {e}")
         raise RuntimeError(f"Failed to resize volume slices. Check input data or dimensions. Original HxW: {current_height}x{current_width}")

    return volume_resized

def get_views(volume, size=(128, 128), depth=64):
    # Assuming input volume is [D, H, W]
    # Resize axial view
    axial = resize_volume(volume, desired_depth=depth, desired_height=size[0], desired_width=size[1])

    # Create other views if needed, applying resize appropriately
    # coronal = resize_volume(np.transpose(volume, (1, 0, 2)), desired_depth=depth, desired_height=size[0], desired_width=size[1])
    # sagittal = resize_volume(np.transpose(volume, (2, 0, 1)), desired_depth=depth, desired_height=size[0], desired_width=size[1])
    coronal = None # Keep None if not needed for classification
    sagittal = None # Keep None if not needed for classification

    return axial, coronal, sagittal