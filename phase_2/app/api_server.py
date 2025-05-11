import os
import torch
import shutil
import tempfile
import zipfile
import time
import traceback
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException # Keep Form for future use if needed
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# --- Import project modules ---
try:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # phase_2
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Project Root added to sys.path: {project_root}")

    from models.cnn_model import PulmonaryInfarction3DCNN
    from src.inference import full_inference_with_explanation
except ImportError as e:
     print(f"ERROR: Failed to import necessary modules: {e}")
     PulmonaryInfarction3DCNN = None
     full_inference_with_explanation = None
except Exception as e:
    print(f"ERROR during initial imports: {e}")
    PulmonaryInfarction3DCNN = None
    full_inference_with_explanation = None

# --- Configuration ---
OUTPUT_FOLDER = "outputs" # Relative to phase_2
UPLOAD_FOLDER = "uploads" # Relative to phase_2
STATIC_FOLDER = "static"  # Relative to phase_2, for background video
MODEL_PATH = "models/pulmonary_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# --- FastAPI App Initialization ---
app = FastAPI(title="Pulmonary Infarction Diagnosis API")

# --- Template Configuration ---
templates = None
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # app directory
    templates_dir = os.path.join(current_script_dir, "templates") # app/templates
    if not os.path.isdir(templates_dir):
        print(f"ERROR: Templates directory not found at: {templates_dir}")
    else:
        templates = Jinja2Templates(directory=templates_dir)
        print(f"DEBUG: Jinja2Templates configured. Search path: {templates.env.loader.searchpath}")
except Exception as e:
    print(f"ERROR setting up Jinja2Templates: {e}")

# --- Static File Serving ---
# For background video etc.
try:
    static_dir_app_path = os.path.join(project_root, STATIC_FOLDER) # phase_2/static
    os.makedirs(static_dir_app_path, exist_ok=True)
    app.mount(f"/{STATIC_FOLDER}", StaticFiles(directory=static_dir_app_path), name=STATIC_FOLDER)
    print(f"Mounted app static directory '{static_dir_app_path}' at '/{STATIC_FOLDER}'")
except Exception as e:
     print(f"ERROR mounting app static directory '{STATIC_FOLDER}': {e}")

# For generated output images
try:
    output_dir_app_path = os.path.join(project_root, OUTPUT_FOLDER) # phase_2/outputs
    os.makedirs(output_dir_app_path, exist_ok=True)
    app.mount(f"/{OUTPUT_FOLDER}", StaticFiles(directory=output_dir_app_path), name=OUTPUT_FOLDER)
    print(f"Mounted output images static directory '{output_dir_app_path}' at '/{OUTPUT_FOLDER}'")
except Exception as e:
     print(f"ERROR mounting output images static directory '{OUTPUT_FOLDER}': {e}")

# --- Model Loading ---
model = None
if PulmonaryInfarction3DCNN:
    model_load_path = os.path.join(project_root, MODEL_PATH) # phase_2/models/pulmonary_model.pth
    print(f"Loading model from: {model_load_path} onto device: {DEVICE}")
    if not os.path.exists(model_load_path):
         print(f"ERROR: Model file not found at {model_load_path}.")
    else:
        try:
            model = PulmonaryInfarction3DCNN(num_classes=NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load model: {type(e).__name__} - {e}"); model = None
else:
     print("ERROR: Model class not imported.")

# --- Ensure Upload Directory Exists ---
upload_dir_abs_path = os.path.join(project_root, UPLOAD_FOLDER)
os.makedirs(upload_dir_abs_path, exist_ok=True)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def show_upload_form(request: Request):
    """Serves the main upload page with video background."""
    if not templates: return HTMLResponse("Server Error: Templates not configured.", status_code=500)
    # Renders phase_2/app/templates/upload.html
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict_patient", response_class=HTMLResponse)
async def handle_prediction(request: Request, file: UploadFile = File(...)):
    """Handles ZIP file upload, runs inference, returns results page."""
    print(f"\nReceived POST request to /predict_patient with file: {file.filename}")
    if not model: return templates.TemplateResponse("result.html", {"request": request, "stage": "Server Error: Model not available.", "explanation":"Model could not be loaded."}, status_code=503)
    if not templates: return HTMLResponse("Server Error: Templates not configured.", status_code=500)
    if not full_inference_with_explanation: return templates.TemplateResponse("result.html", {"request": request, "stage": "Server Error: Inference function not available.", "explanation":"Core processing logic missing."}, status_code=500)

    if not file.filename or not file.filename.lower().endswith(".zip"):
         print(f"Invalid file type: {file.filename}")
         return templates.TemplateResponse("result.html", {
             "request": request, "stage": "Error: Invalid file type. Upload .zip.",
             "explanation":"Please upload a ZIP archive containing DICOM (.dcm) files.",
             "ct_image_url": None, "gradcam_image_url": None, "shap_image_url": None
         }, status_code=400)

    # Use absolute paths for file operations
    upload_dir_abs = os.path.join(project_root, UPLOAD_FOLDER)
    output_dir_abs = os.path.join(project_root, OUTPUT_FOLDER)

    timestamp_process = f"{int(time.time())}_{np.random.randint(1000):03d}"
    base_filename = os.path.splitext(file.filename)[0]
    safe_base_filename = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in base_filename)
    extraction_folder_abs = os.path.join(upload_dir_abs, f"{safe_base_filename}_{timestamp_process}")
    zip_save_path_abs = f"{extraction_folder_abs}.zip"

    stage_result, ct_url, heatmap_url, shap_url = "Error: Processing failed", None, None, None
    explanation_text = "Processing failed before explanation generation."

    try:
        print(f"Saving uploaded ZIP to: {zip_save_path_abs}")
        try:
            with open(zip_save_path_abs, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        except Exception as e_save: raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e_save}")
        finally: await file.close()

        print(f"Extracting ZIP to: {extraction_folder_abs}")
        os.makedirs(extraction_folder_abs, exist_ok=True)
        with zipfile.ZipFile(zip_save_path_abs, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if not any(f.lower().endswith(('.dcm', '.dicom')) for f in file_list if not f.startswith('__MACOSX') and not f.endswith('/')):
                raise ValueError("ZIP archive does not appear to contain any DICOM (.dcm, .dicom) files.")
            zip_ref.extractall(extraction_folder_abs)
        print("Extraction complete.")

        if os.path.exists(zip_save_path_abs): os.remove(zip_save_path_abs); print(f"Removed temp ZIP: {zip_save_path_abs}")

        actual_dicom_folder = extraction_folder_abs
        if not os.listdir(actual_dicom_folder): raise ValueError("Extracted folder is empty.")

        print(f"Running inference on: {actual_dicom_folder}")
        _, stage_result, ct_fname, gradcam_fname, shap_fname = \
            full_inference_with_explanation(model, actual_dicom_folder, output_dir_abs)

        if "Error:" not in stage_result:
             explanation_text = f"The model's prediction is '{stage_result}'. "
             if gradcam_fname and shap_fname: explanation_text += "Grad-CAM highlights model focus. SHAP shows pixel contributions (Red=positive, Blue=negative)."
             elif gradcam_fname: explanation_text += "Grad-CAM highlights model focus."
             elif shap_fname: explanation_text += "SHAP shows pixel contributions (Red=positive, Blue=negative)."
             else: explanation_text += "No visual explanations were generated."
        else: explanation_text = f"Could not generate explanation: {stage_result}"

        ct_url = f"/{OUTPUT_FOLDER}/{ct_fname}" if ct_fname else None
        heatmap_url = f"/{OUTPUT_FOLDER}/{gradcam_fname}" if gradcam_fname else None
        shap_url = f"/{OUTPUT_FOLDER}/{shap_fname}" if shap_fname else None
        print(f"Inference complete. Stage: {stage_result}")

    except HTTPException: raise
    except ValueError as e: stage_result = f"Error: {e}"; explanation_text = f"Processing failed: {e}"; print(f"Processing Error (ValueError): {e}")
    except zipfile.BadZipFile: stage_result = "Error: Invalid ZIP."; explanation_text = "Invalid ZIP."; print(f"Error: Invalid ZIP: {file.filename if file else 'N/A'}")
    except FileNotFoundError as e: stage_result = "Error: Files not found."; explanation_text = f"Processing failed: {e}"; print(f"File Not Found Error: {e}")
    except Exception as e_proc:
        stage_result = "Error: An unexpected error occurred."; explanation_text = "Internal server error."; print(f"ERROR: {type(e_proc).__name__} - {e_proc}"); traceback.print_exc()
    finally:
        if os.path.isdir(extraction_folder_abs): shutil.rmtree(extraction_folder_abs); print(f"Removed extracted folder: {extraction_folder_abs}")

    # Renders phase_2/app/templates/result.html
    return templates.TemplateResponse("result.html", {
        "request": request,
        "stage": stage_result,
        "ct_image_url": ct_url,
        "gradcam_image_url": heatmap_url,
        "shap_image_url": shap_url,
        "explanation": explanation_text
    })