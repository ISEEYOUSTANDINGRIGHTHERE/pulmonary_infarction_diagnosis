import os
import torch
import shutil
import tempfile
import zipfile
import time # For unique identifiers if needed elsewhere
import traceback # For detailed error logging
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse # RedirectResponse not used
from starlette.status import HTTP_303_SEE_OTHER # Not used
import numpy as np # For random int in timestamp if needed elsewhere

# Assuming model and inference are structured as per previous discussions
try:
    # Make sure the current directory (phase_2) is in the path if needed
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # phase_2
    current_working_dir = os.getcwd() # Where uvicorn is launched (should be phase_2)

    # This sys.path manipulation is okay but can be tricky.
    # A better long-term approach is to structure phase_2 as a package
    # or ensure uvicorn is always run from phase_2 and use relative imports.
    if project_root not in sys.path and os.path.basename(current_working_dir) == 'phase_2':
        sys.path.insert(0, project_root)
        print(f"Project Root added to sys.path: {project_root}")
    # This elif condition is a bit confusing and might not always behave as expected
    # if the script isn't in phase_2/app or CWD isn't phase_2.
    # A simpler approach if always running from phase_2:
    # elif os.path.dirname(current_script_dir) not in sys.path: # current_script_dir is 'app'
    # sys.path.insert(0, os.path.dirname(current_script_dir)) # Adds 'phase_2' to path
    # print(f"Parent of 'app' dir added to sys.path: {os.path.dirname(current_script_dir)}")


    from models.cnn_model import PulmonaryInfarction3DCNN
    from src.inference import full_inference_with_explanation
except ImportError as e:
     print(f"ERROR: Failed to import necessary modules: {e}")
     print("Ensure cnn_model.py and inference.py are in the correct locations (models/ and src/).")
     PulmonaryInfarction3DCNN = None
     full_inference_with_explanation = None
except Exception as e:
    print(f"ERROR during initial imports: {e}")
    PulmonaryInfarction3DCNN = None
    full_inference_with_explanation = None


# --- Configuration ---
# These paths are relative to where uvicorn is run (expected to be 'phase_2')
OUTPUT_FOLDER = "outputs"
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/pulmonary_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4 # Example: NoPI, S1, S2, S3

# --- FastAPI App Initialization ---
app = FastAPI(title="Pulmonary Infarction Diagnosis API")

# --- Template Configuration ---
templates = None
try:
    # Assumes api_server.py is in phase_2/app/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_script_dir, "templates") # -> phase_2/app/templates
    if not os.path.isdir(templates_dir):
        print(f"ERROR: Templates directory not found at: {templates_dir}")
        # templates will remain None, handled in routes
    else:
        templates = Jinja2Templates(directory=templates_dir)
        print("-" * 30)
        print(f"DEBUG: Jinja2Templates configured.")
        if hasattr(templates, 'env') and hasattr(templates.env, 'loader') and hasattr(templates.env.loader, 'searchpath'):
            print(f"Resolved search path: {templates.env.loader.searchpath}")
        print(f"Current Working Directory for template loading context: {os.getcwd()}")
        print("-" * 30)
except Exception as e:
    print(f"ERROR setting up Jinja2Templates: {e}")
    # templates remains None

# --- Static File Serving ---
try:
    # Assuming uvicorn is run from 'phase_2' directory
    # OUTPUT_FOLDER is "outputs", so static_dir_path becomes "phase_2/outputs"
    static_dir_path = OUTPUT_FOLDER # Already relative to CWD if uvicorn run from phase_2
    os.makedirs(static_dir_path, exist_ok=True)
    app.mount(f"/{OUTPUT_FOLDER}", StaticFiles(directory=static_dir_path), name=OUTPUT_FOLDER)
    print(f"Mounted static directory '{static_dir_path}' at '/{OUTPUT_FOLDER}'")
except Exception as e:
     print(f"ERROR mounting static directory '{OUTPUT_FOLDER}': {e}")

# --- Model Loading ---
model = None
if PulmonaryInfarction3DCNN: # Check if import succeeded
    model_load_path = MODEL_PATH # Already relative to CWD if uvicorn run from phase_2
    print(f"Loading model from: {model_load_path} onto device: {DEVICE}")
    if not os.path.exists(model_load_path):
         print(f"ERROR: Model file not found at {model_load_path}. API cannot make predictions.")
    else:
        try:
            model = PulmonaryInfarction3DCNN(num_classes=NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
            model.eval()
            print("Model loaded successfully.")
        except FileNotFoundError: # Should be caught by os.path.exists
            print(f"ERROR: Model file definitely not found at {model_load_path} during load attempt.")
            model = None
        except Exception as e:
            print(f"ERROR: Failed to load model state_dict: {type(e).__name__} - {e}")
            traceback.print_exc()
            model = None
else:
     print("ERROR: Model class PulmonaryInfarction3DCNN not imported, cannot load model.")

# --- Ensure Upload Directory Exists ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Relative to CWD (phase_2)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main upload page."""
    if not templates:
         return HTMLResponse("Server Configuration Error: Templates not loaded.", status_code=500)
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict_patient", response_class=HTMLResponse)
async def predict_patient(request: Request, file: UploadFile = File(...)):
    """Handles ZIP file upload containing DICOMs, runs inference, returns results."""
    print(f"\nReceived request for /predict_patient with file: {file.filename}")
    if not model: return HTMLResponse("Server Error: Model not available.", status_code=503)
    if not templates: return HTMLResponse("Server Configuration Error: Templates not loaded.", status_code=500)
    if not full_inference_with_explanation: return HTMLResponse("Server Configuration Error: Inference function not available.", status_code=500)

    if not file.filename or not file.filename.lower().endswith(".zip"):
         print(f"Invalid file type: {file.filename}")
         return templates.TemplateResponse("result.html", {
             "request": request, "stage": "Error: Invalid file type. Please upload a .zip file.",
             "ct_image_url": None, "gradcam_image_url": None, "shap_image_url": None, "explanation": None
         }, status_code=400)

    # Define paths relative to phase_2 (where uvicorn runs)
    upload_dir_rel = UPLOAD_FOLDER
    output_dir_rel = OUTPUT_FOLDER # Where inference function saves images

    # Unique temporary paths for this request
    timestamp_process = f"{int(time.time())}_{np.random.randint(1000):03d}"
    base_filename = os.path.splitext(file.filename)[0]
    safe_base_filename = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in base_filename)
    extraction_folder_rel = os.path.join(upload_dir_rel, f"{safe_base_filename}_{timestamp_process}")
    zip_save_path_rel = f"{extraction_folder_rel}.zip" # Place zip alongside temp folder

    # Use absolute paths for most file operations to avoid CWD ambiguity
    extraction_folder_abs = os.path.abspath(extraction_folder_rel)
    output_dir_abs = os.path.abspath(output_dir_rel)
    zip_save_path_abs = os.path.abspath(zip_save_path_rel)

    # Initialize context variables for template, especially explanation_text
    stage_result = "Error: Processing failed"
    pred_result = -1
    ct_url, heatmap_url, shap_url = None, None, None
    explanation_text = "Processing failed before explanation generation." # Default explanation

    try:
        # --- 1. Save and Extract Uploaded ZIP ---
        # Ensure parent of zip_save_path_abs exists (which is 'uploads' directory)
        os.makedirs(os.path.dirname(zip_save_path_abs), exist_ok=True)
        print(f"Saving uploaded ZIP to: {zip_save_path_abs}")
        try:
            with open(zip_save_path_abs, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e_save:
            print(f"Error saving uploaded file: {e_save}")
            # Re-raise as HTTPException to be caught by FastAPI's error handling for a clean response
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e_save}")
        finally:
            await file.close() # Close file handle of UploadFile

        print(f"Extracting ZIP to: {extraction_folder_abs}")
        os.makedirs(extraction_folder_abs, exist_ok=True) # Ensure extraction target exists
        with zipfile.ZipFile(zip_save_path_abs, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if not any(f.lower().endswith(('.dcm', '.dicom')) for f in file_list if not f.startswith('__MACOSX') and not f.endswith('/')):
                raise ValueError("ZIP archive does not appear to contain any DICOM (.dcm, .dicom) files.")
            zip_ref.extractall(extraction_folder_abs)
        print("Extraction complete.")

        # Clean up the ZIP file now that extraction is done
        if os.path.exists(zip_save_path_abs):
            print(f"Removing temporary ZIP file: {zip_save_path_abs}")
            os.remove(zip_save_path_abs)

        # Check if extraction folder actually contains usable files (more specific check)
        dicom_files_found = False
        for root, _, files in os.walk(extraction_folder_abs):
             if any(f.lower().endswith(('.dcm', '.dicom')) for f in files):
                 # Pass the directory that contains the .dcm files to inference
                 # This handles DICOMs being in a subfolder within the zip
                 actual_dicom_folder = root
                 dicom_files_found = True
                 break
        if not dicom_files_found:
             raise ValueError("Extracted folder is empty or no DICOM files found. Ensure ZIP contains DICOM files directly or in one subfolder.")


        # --- 2. Run Inference ---
        print(f"Running inference on extracted DICOM folder: {actual_dicom_folder}")
        pred_result, stage_result, ct_fname, gradcam_fname, shap_fname = \
            full_inference_with_explanation(
                model=model,
                dicom_folder=actual_dicom_folder, # Pass the folder that contains DICOMs
                output_folder=output_dir_abs
            )

        # --- 3. Generate Explanation Text ---
        if "Error:" not in stage_result:
             explanation_text = f"The model's prediction is '{stage_result}'. "
             if gradcam_fname and shap_fname:
                 explanation_text += "Grad-CAM highlights the general input regions the model focused on. SHAP shows the specific contribution of each pixel (Red increases likelihood of prediction, Blue decreases)."
             elif gradcam_fname:
                 explanation_text += "Grad-CAM highlights the general input regions the model focused on for this prediction."
             elif shap_fname:
                 explanation_text += "SHAP shows the contribution of each pixel (Red increases likelihood of prediction, Blue decreases)."
             else:
                 explanation_text += "No visual explanations were successfully generated for this prediction."
        else:
             explanation_text = f"Could not generate explanation: {stage_result}"


        # --- 4. Convert Filenames to URL Paths (relative to OUTPUT_FOLDER) ---
        ct_url = f"/{OUTPUT_FOLDER}/{ct_fname}" if ct_fname else None
        heatmap_url = f"/{OUTPUT_FOLDER}/{gradcam_fname}" if gradcam_fname else None
        shap_url = f"/{OUTPUT_FOLDER}/{shap_fname}" if shap_fname else None
        print(f"Inference processing complete. Stage: {stage_result}")

    # --- Handle Errors During Processing ---
    except ValueError as e_val: # Catch specific errors like empty zip/folder
        print(f"Processing Error (ValueError): {e_val}")
        stage_result = f"Error: {e_val}"
        explanation_text = f"Processing failed: {e_val}"
    except zipfile.BadZipFile:
        print(f"Error: Uploaded file is not a valid ZIP archive: {file.filename if file else 'N/A'}")
        stage_result = "Error: Invalid ZIP file."
        explanation_text = "Uploaded file was not a valid ZIP archive."
    except FileNotFoundError as e_fnf: # Catch issues finding folders/files
        print(f"File Not Found Error during processing: {e_fnf}")
        stage_result = "Error: Could not find necessary files during processing."
        explanation_text = f"Processing failed: {e_fnf}"
    except HTTPException: # Re-raise HTTPExceptions to let FastAPI handle them
        raise
    except Exception as e_proc: # Catch all other errors
        print(f"ERROR during processing/inference call: {type(e_proc).__name__} - {e_proc}")
        traceback.print_exc()
        stage_result = "Error: An unexpected error occurred during diagnosis."
        explanation_text = "An internal server error prevented explanation generation."
    finally:
        # --- ALWAYS Clean up the extracted DICOM folder ---
        try:
            if os.path.isdir(extraction_folder_abs): # Check if it was created
                 print(f"Removing extracted folder: {extraction_folder_abs}")
                 shutil.rmtree(extraction_folder_abs)
        except Exception as e_clean_dir:
            print(f"Error removing extracted folder {extraction_folder_abs}: {e_clean_dir}")
        # --- ---

    # --- 5. Render results page ---
    # Renders phase_2/app/templates/result.html
    return templates.TemplateResponse("result.html", {
        "request": request,
        "stage": stage_result,
        "ct_image_url": ct_url,
        "gradcam_image_url": heatmap_url,
        "shap_image_url": shap_url,
        "explanation": explanation_text
    })