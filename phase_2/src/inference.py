# src/inference.py

import matplotlib
matplotlib.use('Agg') # Keep this at the TOP!
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import cv2
import time
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: 'shap' library not installed. SHAP explanations will be skipped.")

from src.utils.dicom_loader import load_dicom_volume, get_views
from src.xai.grad_cam import GradCAM # Uses the modified GradCAM class now
import torch.nn.functional as F # Make sure F is imported if used in GradCAM class

# --- generate_gradcam function ---
# (Assumed correct from previous step - uses updated GradCAM class)
def generate_gradcam(model, input_tensor, target_class):
    print("Generating Grad-CAM...")
    try:
        # <<<!!! ENSURE THIS IS THE CORRECT FINAL CONV LAYER FOR YOUR MODEL !!!>>>
        target_layer = model.conv3
        # <<<!!! ========================================================== !!!>>>
        if target_layer is None or not isinstance(target_layer, torch.nn.Module):
             raise ValueError(f"Target layer specified for Grad-CAM is invalid or not found.")
    except AttributeError as e:
        print(f"ERROR: Could not access the specified target layer for Grad-CAM: {e}")
        raise AttributeError(f"Target layer for Grad-CAM not found or accessible: {e}")

    model.eval()
    gradcam = GradCAM(model=model, target_layer=target_layer)
    heatmap = gradcam.generate(input_tensor=input_tensor.clone().requires_grad_(True), target_class=target_class)
    gradcam.remove_hooks()
    if heatmap is None or heatmap.size == 0 or not np.any(heatmap):
        print("Grad-CAM heatmap generation failed or result was empty/zero.")
        return None
    print(f"Grad-CAM heatmap generated (shape: {heatmap.shape}, min: {heatmap.min():.2f}, max: {heatmap.max():.2f}).")
    return heatmap

# --- Label Map ---
LABEL_MAP_INV = {
    0: "Absence of Pulmonary Infarction",
    1: "Pulmonary Infarction: Stage 1",
    2: "Pulmonary Infarction: Stage 2",
    3: "Pulmonary Infarction: Stage 3"
}

# --- Main Inference Function ---
def full_inference_with_explanation(model, dicom_folder, output_folder):
    model.eval()
    device = next(model.parameters()).device
    timestamp = f"{int(time.time())}_{np.random.randint(1000):03d}"
    print(f"Running inference for folder: {dicom_folder} (Timestamp: {timestamp})")

    ct_filename = None
    gradcam_filename = None
    shap_filename = None
    pred = -1
    stage = "Error: Processing failed"

    try:
        # --- 1. Load Data ---
        print("Loading DICOM volume...")
        volume = load_dicom_volume(dicom_folder)
        axial, _, _ = get_views(volume, size=(128, 128), depth=64) # MATCH TRAINING PARAMS
        axial_tensor = torch.tensor(axial, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        print(f"Data loaded and preprocessed, tensor shape: {axial_tensor.shape}")

        # --- 2. Get Prediction ---
        print("Performing model prediction...")
        with torch.no_grad():
            outputs = model(axial_tensor)
            if not isinstance(outputs, torch.Tensor) or outputs.ndim < 2:
                 raise ValueError(f"Model output has unexpected shape or type: {outputs.shape}")
            pred = torch.argmax(outputs, dim=1).item()
            try: stage = LABEL_MAP_INV[pred]
            except KeyError: stage = f"Unknown Class {pred}"
        print(f"Prediction done. Index: {pred}, Stage: {stage}")

        # --- 3. Prepare & Save Base CT Slice Image ---
        display_slice_index = axial_tensor.shape[2] // 2
        slice_np = axial_tensor[0, 0, display_slice_index].cpu().numpy().astype(np.float32)
        slice_display = cv2.normalize(slice_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ct_filename = f"ct_slice_{timestamp}.png"
        ct_save_path = os.path.join(output_folder, ct_filename)
        success_ct = cv2.imwrite(ct_save_path, slice_display)
        if success_ct: print(f"Saved base CT slice to: {ct_save_path}")
        else: print(f"Error saving base CT slice."); ct_filename = None

        # --- 4. Generate & Save Grad-CAM Image ---
        try:
            heatmap_np = generate_gradcam(model, axial_tensor, target_class=pred)
            if heatmap_np is not None:
                 if np.any(heatmap_np):
                     if heatmap_np.shape != slice_display.shape:
                         heatmap_resized = cv2.resize(heatmap_np, (slice_display.shape[1], slice_display.shape[0]), interpolation=cv2.INTER_LINEAR)
                     else: heatmap_resized = heatmap_np
                     h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
                     if h_max - h_min > 1e-8: heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min)
                     else: heatmap_norm = np.zeros_like(heatmap_resized)
                     heatmap_jet = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
                     slice_bgr = cv2.cvtColor(slice_display, cv2.COLOR_GRAY2BGR)
                     alpha = 0.5
                     gradcam_overlay = cv2.addWeighted(slice_bgr, 1 - alpha, heatmap_jet, alpha, 0)
                     gradcam_filename_temp = f"gradcam_{timestamp}.png"
                     gradcam_save_path_temp = os.path.join(output_folder, gradcam_filename_temp)
                     success_gc = cv2.imwrite(gradcam_save_path_temp, gradcam_overlay)
                     if success_gc: gradcam_filename = gradcam_filename_temp; print(f"Saved Grad-CAM overlay to: {gradcam_save_path_temp}")
                     else: print(f"Error saving Grad-CAM overlay image."); gradcam_filename = None
                 else: print("Grad-CAM heatmap was all zeros. Skipping overlay."); gradcam_filename = None
            else: print("Grad-CAM generation returned None. Skipping overlay."); gradcam_filename = None
        except Exception as e_gradcam: print(f"ERROR during Grad-CAM: {type(e_gradcam).__name__} - {e_gradcam}"); gradcam_filename = None

        # --- 5. Generate SHAP Explanation ---
        shap_filename = None
        if SHAP_AVAILABLE:
            try:
                print("Attempting SHAP explanation (this might take time)...")
                start_shap = time.time()
                model.eval()

                # --- 1. Background Data ---
                background_samples = 5
                background = torch.zeros(background_samples, *axial_tensor.shape[1:]).to(device)
                print(f"Using SHAP background data shape: {background.shape}")

                # --- 2. Create Explainer ---
                explainer = shap.GradientExplainer(model, background)
                print("SHAP GradientExplainer created.")

                # --- 3. Compute SHAP values ---
                num_shap_samples = 10 #can be adjusted but for a 5d sensor volume 50, is what we should be using. 
                print(f"Calculating SHAP values for input shape {axial_tensor.shape} using nsamples={num_shap_samples}...")
                shap_values_output = explainer.shap_values(axial_tensor, nsamples=num_shap_samples)
                print(f"SHAP values calculated. Output type: {type(shap_values_output)}") # Will be <class 'numpy.ndarray'>

                # --- 4. Select SHAP values for predicted class ---
                # Check if it's a list (multi-class output)
                if isinstance(shap_values_output, list) and len(shap_values_output) > pred:
                     shap_values_for_pred = shap_values_output[pred] # Should be numpy array
                     print(f"Selected SHAP values for predicted class {pred}.")
                elif isinstance(shap_values_output, np.ndarray):
                     # If already numpy, assume it's for the only/relevant class or needs class indexing
                     print("Warning: SHAP output was NumPy array, not list. Assuming it's for predicted class or needs indexing.")
                     # Check if it has a class dimension - assumes shape [N, Class, C, D, H, W] or similar if multi-output numpy
                     if shap_values_output.ndim == axial_tensor.ndim + 1 and shap_values_output.shape[1] > pred : # Check if class dim exists
                          shap_values_for_pred = shap_values_output[:, pred, ...] # Example if class is dim 1
                     else:
                          shap_values_for_pred = shap_values_output # Use as is, might be [N, C, D, H, W]
                else:
                     raise TypeError(f"Unexpected SHAP output type: {type(shap_values_output)}")

                # --- 5. Prepare Slices for Visualization ---
                print(f"SHAP values for pred shape before slicing: {shap_values_for_pred.shape}")
                # *** THE FIX IS HERE: REMOVE .cpu().numpy() *** if we face any issues with the SHAP values.
                shap_slice_np = shap_values_for_pred[0, 0, display_slice_index]
                # *** END FIX ***
                print(f"Extracted SHAP slice shape: {shap_slice_np.shape}")

                # --- 6. Create Overlay Plot ---
                fig_shap, ax_shap = plt.subplots(figsize=(6, 6))
                max_abs_shap = np.percentile(np.abs(shap_slice_np), 99.9)
                if max_abs_shap < 1e-9: max_abs_shap = 1e-9
                shap_norm = shap_slice_np / max_abs_shap
                cmap_shap = plt.cm.bwr
                ax_shap.imshow(slice_display, cmap='gray', aspect='auto')
                im_shap = ax_shap.imshow(shap_norm, cmap=cmap_shap, alpha=0.55, vmin=-1, vmax=1, aspect='auto')
                fig_shap.colorbar(im_shap, ax=ax_shap, label='Normalized SHAP Value Contribution')
                ax_shap.set_title(f'SHAP Values (Contribution to "{stage}")')
                ax_shap.axis('off')

                # --- 7. Save the plot ---
                shap_filename_temp = f"shap_{timestamp}.png"
                shap_save_path_temp = os.path.join(output_folder, shap_filename_temp)
                fig_shap.savefig(shap_save_path_temp, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig_shap)
                shap_filename = shap_filename_temp
                end_shap = time.time()
                print(f"Saved SHAP plot to: {shap_save_path_temp} (took {end_shap - start_shap:.2f}s)")

            except Exception as e_shap:
                print(f"ERROR during SHAP generation/saving: {type(e_shap).__name__} - {e_shap}")
                import traceback; traceback.print_exc()
                shap_filename = None
        else:
             print("SHAP library not installed (SHAP_AVAILABLE=False), skipping SHAP explanation.")

    # Error handling for main processing
    except ValueError as e_load: stage = f"Error: {e_load}"; print(f"ERROR: {e_load}")
    except FileNotFoundError as e_load_fnf: stage = "Error: Input folder not found"; print(f"ERROR: Input folder not found: {dicom_folder}")
    except Exception as e_main:
        stage = "Error: Inference Failed"; print(f"ERROR: {type(e_main).__name__} - {e_main}"); traceback.print_exc()
        ct_filename = ct_filename; gradcam_filename = gradcam_filename; shap_filename = shap_filename

    # --- 6. Return Results ---
    print(f"Returning: pred={pred}, stage='{stage}', ct='{ct_filename}', gradcam='{gradcam_filename}', shap='{shap_filename}'")
    return (pred, stage, ct_filename, gradcam_filename, shap_filename)