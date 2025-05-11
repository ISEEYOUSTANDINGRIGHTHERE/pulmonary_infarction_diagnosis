import os
import torch
import shutil
import tempfile
import zipfile
import time
import traceback
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# --- Import project modules ---
try:
    import sys
    # Assuming api_server.py is in PULMONARY_INFARCTION_DIAGNOSIS/phase_2/app/
    current_script_dir_for_project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir_for_project_root) # phase_2 directory
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Project Root added to sys.path: {project_root}")

    from models.cnn_model import PulmonaryInfarction3DCNN
    from src.inference import full_inference_with_explanation
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    traceback.print_exc()
    PulmonaryInfarction3DCNN = None
    full_inference_with_explanation = None
except Exception as e:
    print(f"ERROR during initial imports: {e}")
    traceback.print_exc()
    PulmonaryInfarction3DCNN = None
    full_inference_with_explanation = None

# --- Configuration ---
OUTPUT_FOLDER_NAME = "outputs" # Name of the folder for generated images, relative to project_root
UPLOAD_FOLDER_NAME = "uploads" # Name of the folder for uploads, relative to project_root
MODEL_PATH_RELATIVE_TO_PROJECT_ROOT = "models/pulmonary_model.pth" # Relative to project_root
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# --- FastAPI App Initialization ---
app = FastAPI(title="Pulmonary Infarction Diagnosis API")

# --- Template Configuration ---
templates = None
try:
    # api_server.py is in phase_2/app/
    current_app_dir_for_templates = os.path.dirname(os.path.abspath(__file__)) # phase_2/app/
    templates_dir = os.path.join(current_app_dir_for_templates, "templates")   # phase_2/app/templates/
    if not os.path.isdir(templates_dir):
        print(f"ERROR: Templates directory not found at: {templates_dir}")
    else:
        templates = Jinja2Templates(directory=templates_dir)
        print(f"DEBUG: Jinja2Templates configured. Search path: {templates.env.loader.searchpath if hasattr(templates.env.loader, 'searchpath') else 'Default'}")
except Exception as e:
    print(f"ERROR setting up Jinja2Templates: {e}")
    traceback.print_exc()

# --- Static File Serving ---

# 1. Static files for the application's templates (e.g., background video in upload.html)
# These are located in a 'static' folder within the 'app' directory.
APP_STATIC_MOUNT_URL = "/static" # URL path to access these files
APP_STATIC_DIR_NAME = "static"   # Name of the directory inside the app folder (e.g., app/static)
APP_MOUNT_NAME_FOR_URL_FOR = "static" # Name used in url_for() in templates

try:
    # __file__ is api_server.py, which is in phase_2/app/
    current_app_dir = os.path.dirname(os.path.abspath(__file__))
    # This will correctly resolve to: .../phase_2/app/static/
    app_specific_static_dir = os.path.join(current_app_dir, APP_STATIC_DIR_NAME)

    if not os.path.isdir(app_specific_static_dir):
        print(f"WARNING: Application static directory '{app_specific_static_dir}' not found. Creating it.")
        os.makedirs(app_specific_static_dir, exist_ok=True)

    print(f"Attempting to mount application static directory: '{app_specific_static_dir}' at URL '{APP_STATIC_MOUNT_URL}'")
    app.mount(APP_STATIC_MOUNT_URL, StaticFiles(directory=app_specific_static_dir), name=APP_MOUNT_NAME_FOR_URL_FOR)
    print(f"SUCCESS: Mounted application static directory '{app_specific_static_dir}' at URL '{APP_STATIC_MOUNT_URL}' with mount name '{APP_MOUNT_NAME_FOR_URL_FOR}'.")
    print(f"Ensure 'background-video.mp4' is in this directory: '{app_specific_static_dir}'.")

except Exception as e:
    print(f"ERROR mounting application static directory '{APP_STATIC_DIR_NAME}': {type(e).__name__} - {e}")
    traceback.print_exc()

# 2. Static serving for generated output images (e.g., CT scans, Grad-CAM)
# These are in project_root/outputs/
OUTPUT_FILES_MOUNT_URL = f"/{OUTPUT_FOLDER_NAME}" # URL path, e.g., /outputs
try:
    # project_root should be 'phase_2/'
    output_dir_abs_path = os.path.join(project_root, OUTPUT_FOLDER_NAME) # e.g., phase_2/outputs
    os.makedirs(output_dir_abs_path, exist_ok=True)
    app.mount(OUTPUT_FILES_MOUNT_URL, StaticFiles(directory=output_dir_abs_path), name=OUTPUT_FOLDER_NAME)
    print(f"SUCCESS: Mounted output images static directory '{output_dir_abs_path}' at URL '{OUTPUT_FILES_MOUNT_URL}'")
except Exception as e:
    print(f"ERROR mounting output images static directory '{OUTPUT_FOLDER_NAME}': {type(e).__name__} - {e}")
    traceback.print_exc()


# --- Model Loading ---
model = None
if PulmonaryInfarction3DCNN:
    model_load_path = os.path.join(project_root, MODEL_PATH_RELATIVE_TO_PROJECT_ROOT)
    print(f"Attempting to load model from: {model_load_path} onto device: {DEVICE}")
    if not os.path.exists(model_load_path):
        print(f"ERROR: Model file not found at {model_load_path}.")
    else:
        try:
            model = PulmonaryInfarction3DCNN(num_classes=NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load model: {type(e).__name__} - {e}"); traceback.print_exc(); model = None
else:
    print("ERROR: Model class 'PulmonaryInfarction3DCNN' not imported correctly. Model cannot be loaded.")

# --- Ensure Upload Directory Exists (relative to project_root) ---
upload_dir_abs_path = os.path.join(project_root, UPLOAD_FOLDER_NAME)
os.makedirs(upload_dir_abs_path, exist_ok=True)
print(f"Upload directory ensured at: {upload_dir_abs_path}")


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def show_upload_form(request: Request):
    """Serves the main upload page."""
    if not templates:
        return HTMLResponse("Server Error: Templates not configured. Check server logs.", status_code=500)
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict_patient", response_class=HTMLResponse)
async def handle_prediction(request: Request, file: UploadFile = File(...)):
    """Handles ZIP file upload, runs inference, returns results page."""
    print(f"\nReceived POST request to /predict_patient with file: {file.filename}")

    if not model:
        error_message = "Server Error: Model not available. Check server logs for loading issues."
        print(error_message)
        return templates.TemplateResponse("result.html", {"request": request, "stage": error_message, "explanation": "The prediction model could not be loaded."}, status_code=503)
    if not templates:
        error_message = "Server Error: Templates not configured. Check server logs."
        print(error_message)
        return HTMLResponse(error_message, status_code=500)
    if not full_inference_with_explanation:
        error_message = "Server Error: Inference function not available. Core processing logic missing. Check imports."
        print(error_message)
        return templates.TemplateResponse("result.html", {"request": request, "stage": error_message, "explanation": "The core inference function is missing."}, status_code=500)

    if not file.filename or not file.filename.lower().endswith(".zip"):
        print(f"Invalid file type: {file.filename}")
        return templates.TemplateResponse("result.html", {
            "request": request, "stage": "Error: Invalid file type. Please upload a .zip archive.",
            "explanation": "The uploaded file must be a ZIP archive (.zip) containing DICOM (.dcm) files.",
            "ct_image_url": None, "gradcam_image_url": None, "shap_image_url": None
        }, status_code=400)

    output_dir_abs_for_inference = os.path.join(project_root, OUTPUT_FOLDER_NAME)
    timestamp_process = f"{int(time.time())}_{np.random.randint(1000):03d}"
    base_filename = os.path.splitext(file.filename)[0]
    safe_base_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_filename)
    extraction_folder_name = f"{safe_base_filename}_{timestamp_process}"
    extraction_folder_abs = os.path.join(upload_dir_abs_path, extraction_folder_name)
    zip_save_path_abs = os.path.join(upload_dir_abs_path, f"{extraction_folder_name}.zip")

    stage_result, ct_url, heatmap_url, shap_url = "Error: Processing failed unexpectedly.", None, None, None
    explanation_text = "Processing failed before detailed explanation could be generated."

    try:
        print(f"Attempting to save uploaded ZIP to: {zip_save_path_abs}")
        try:
            with open(zip_save_path_abs, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"Successfully saved ZIP to: {zip_save_path_abs}")
        except Exception as e_save:
            print(f"ERROR saving uploaded file: {type(e_save).__name__} - {e_save}"); traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e_save}")
        finally:
            await file.close()

        print(f"Attempting to extract ZIP from '{zip_save_path_abs}' to: {extraction_folder_abs}")
        os.makedirs(extraction_folder_abs, exist_ok=True)
        with zipfile.ZipFile(zip_save_path_abs, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            dicom_files_in_zip = [f for f in file_list if f.lower().endswith(('.dcm', '.dicom')) and not f.startswith('__MACOSX') and not f.endswith('/')]
            if not dicom_files_in_zip:
                print("ERROR: ZIP archive does not appear to contain any DICOM (.dcm, .dicom) files.")
                raise ValueError("ZIP archive must contain DICOM (.dcm, .dicom) files directly or within a single subfolder.")
            print(f"DICOM files found in ZIP: {dicom_files_in_zip}")
            zip_ref.extractall(extraction_folder_abs)
        print(f"Extraction complete to: {extraction_folder_abs}")

        if os.path.exists(zip_save_path_abs):
            os.remove(zip_save_path_abs)
            print(f"Removed temporary ZIP: {zip_save_path_abs}")

        actual_dicom_folder_for_inference = extraction_folder_abs
        items_in_extraction = os.listdir(extraction_folder_abs)
        if len(items_in_extraction) == 1 and os.path.isdir(os.path.join(extraction_folder_abs, items_in_extraction[0])):
            potential_inner_folder = os.path.join(extraction_folder_abs, items_in_extraction[0])
            files_at_top_level = [f for f in os.listdir(extraction_folder_abs) if os.path.isfile(os.path.join(extraction_folder_abs, f))]
            if not files_at_top_level:
                if any(f.lower().endswith(('.dcm', '.dicom')) for f in os.listdir(potential_inner_folder)):
                    actual_dicom_folder_for_inference = potential_inner_folder
                    print(f"Adjusted DICOM folder for inference to subfolder: {actual_dicom_folder_for_inference}")

        if not os.listdir(actual_dicom_folder_for_inference) or not any(f.lower().endswith(('.dcm', '.dicom')) for f in os.listdir(actual_dicom_folder_for_inference)):
            print(f"ERROR: No DICOM files found in the target processing folder: {actual_dicom_folder_for_inference}")
            raise ValueError("Extracted folder is empty or does not contain DICOM files for processing.")

        print(f"Running inference on DICOM files in: {actual_dicom_folder_for_inference}")
        print(f"Outputting generated images to: {output_dir_abs_for_inference}")

        _, stage_result, ct_fname, gradcam_fname, shap_fname = \
            full_inference_with_explanation(model, actual_dicom_folder_for_inference, output_dir_abs_for_inference, DEVICE)

        if "Error:" not in stage_result and stage_result is not None:
            explanation_text = f"The model's prediction is '{stage_result}'. "
            if gradcam_fname and shap_fname: explanation_text += "Grad-CAM highlights areas the model focused on. SHAP values indicate pixel-level contributions to the prediction (Red=positive influence, Blue=negative influence)."
            elif gradcam_fname: explanation_text += "Grad-CAM highlights areas the model focused on for its prediction."
            elif shap_fname: explanation_text += "SHAP values indicate pixel-level contributions (Red=positive influence, Blue=negative influence)."
            else: explanation_text += "No visual explanations were generated for this prediction."
        elif stage_result is None:
            stage_result = "Error: Prediction result was inconclusive."
            explanation_text = "The model did not return a definitive prediction stage."
        else:
            explanation_text = f"Could not generate a full explanation due to an error during prediction: {stage_result}"

        ct_url = f"{OUTPUT_FILES_MOUNT_URL}/{ct_fname}" if ct_fname else None
        heatmap_url = f"{OUTPUT_FILES_MOUNT_URL}/{gradcam_fname}" if gradcam_fname else None
        shap_url = f"{OUTPUT_FILES_MOUNT_URL}/{shap_fname}" if shap_fname else None
        print(f"Inference complete. Prediction Stage: {stage_result}")
        print(f"CT Image URL: {ct_url}, Grad-CAM URL: {heatmap_url}, SHAP URL: {shap_url}")

    except HTTPException:
        raise
    except ValueError as e_val:
        stage_result = f"Error: {e_val}"
        explanation_text = f"Input data validation failed: {e_val}"
        print(f"Processing Error (ValueError): {e_val}"); traceback.print_exc()
    except zipfile.BadZipFile:
        stage_result = "Error: Invalid or corrupted ZIP file."
        explanation_text = "The uploaded .zip file could not be read. It might be corrupted or not a valid ZIP format."
        print(f"Error: Invalid ZIP file: {file.filename if file else 'N/A'}")
    except FileNotFoundError as e_fnf:
        stage_result = "Error: A required file or directory was not found during processing."
        explanation_text = f"File system error: {e_fnf}"
        print(f"File Not Found Error during processing: {e_fnf}"); traceback.print_exc()
    except Exception as e_proc:
        stage_result = "Error: An unexpected server error occurred during processing."
        explanation_text = "An internal server error prevented the completion of the diagnosis. Please try again later or contact support."
        print(f"Unexpected ERROR during processing: {type(e_proc).__name__} - {e_proc}"); traceback.print_exc()
    finally:
        if os.path.isdir(extraction_folder_abs):
            try:
                shutil.rmtree(extraction_folder_abs)
                print(f"Successfully removed extracted folder: {extraction_folder_abs}")
            except Exception as e_clean:
                print(f"ERROR cleaning up extracted folder '{extraction_folder_abs}': {type(e_clean).__name__} - {e_clean}"); traceback.print_exc()
        if os.path.exists(zip_save_path_abs):
            try:
                os.remove(zip_save_path_abs)
                print(f"Successfully removed temp ZIP file during final cleanup: {zip_save_path_abs}")
            except Exception as e_zip_clean:
                print(f"ERROR cleaning up temp ZIP '{zip_save_path_abs}': {type(e_zip_clean).__name__} - {e_zip_clean}")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "stage": stage_result,
        "ct_image_url": ct_url,
        "gradcam_image_url": heatmap_url,
        "shap_image_url": shap_url,
        "explanation": explanation_text
    })