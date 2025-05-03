import os
import torch
import shutil
import tempfile
import zipfile # To handle ZIP uploads
import time # For unique identifiers if needed elsewhere
import traceback # For detailed error logging
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER
import numpy as np # For random int in timestamp if needed elsewhere

# Assuming model and inference are structured as per previous discussions
try:
    # Make sure the current directory (phase_2) is in the path if needed
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Check if running directly vs via uvicorn might change expected root
    current_working_dir = os.getcwd()
    if project_root not in sys.path and os.path.basename(current_working_dir) == 'phase_2':
         # Add project root if running from phase_2 dir
         sys.path.insert(0, project_root)
         print(f"Project Root added to sys.path: {project_root}")
    elif os.path.dirname(current_working_dir) not in sys.path:
         # Add parent of 'app' dir if running from elsewhere perhaps
         sys.path.insert(0, os.path.dirname(current_working_dir))
         print(f"Parent of 'app' dir added to sys.path: {os.path.dirname(current_working_dir)}")


    from models.cnn_model import PulmonaryInfarction3DCNN
    from src.inference import full_inference_with_explanation
except ImportError as e:
     print(f"ERROR: Failed to import necessary modules: {e}")
     print("Ensure cnn_model.py and inference.py are in the correct locations (models/ and src/).")
     PulmonaryInfarction3DCNN = None
     full_inference_with_explanation = None

# --- Configuration ---
OUTPUT_FOLDER = "outputs"
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/pulmonary_model.pth" # Corrected folder name
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4 # Example: NoPI, S1, S2, S3

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Template Configuration ---
templates = None
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_script_dir, "templates")
    templates = Jinja2Templates(directory=templates_dir)
    print("-" * 30)
    print(f"DEBUG: Jinja2Templates configured.")
    if hasattr(templates, 'env') and hasattr(templates.env, 'loader') and hasattr(templates.env.loader, 'searchpath'):
        print(f"Resolved search path: {templates.env.loader.searchpath}")
    else:
        print("Could not access Jinja2 environment search path for debugging.")
    print(f"Current Working Directory: {os.getcwd()}")
    print("-" * 30)
except Exception as e:
    print(f"ERROR setting up Jinja2Templates: {e}")


# --- Static File Serving ---
try:
    # Ensure OUTPUT_FOLDER path is relative to where uvicorn runs (phase_2)
    static_dir_path = os.path.join(os.getcwd(), OUTPUT_FOLDER) if os.path.basename(os.getcwd()) == 'phase_2' else OUTPUT_FOLDER
    os.makedirs(static_dir_path, exist_ok=True)
    app.mount(f"/{OUTPUT_FOLDER}", StaticFiles(directory=static_dir_path), name=OUTPUT_FOLDER)
    print(f"Mounted static directory '{static_dir_path}' at '/{OUTPUT_FOLDER}'")
except Exception as e:
     print(f"ERROR mounting static directory '{OUTPUT_FOLDER}': {e}")

# --- Model Loading ---
model = None
if PulmonaryInfarction3DCNN: # Check if import succeeded
    # Ensure MODEL_PATH is relative to where uvicorn runs (phase_2)
    model_load_path = os.path.join(os.getcwd(), MODEL_PATH) if os.path.basename(os.getcwd()) == 'phase_2' else MODEL_PATH
    print(f"Loading model from: {model_load_path} onto device: {DEVICE}")
    try:
        model = PulmonaryInfarction3DCNN(num_classes=NUM_CLASSES).to(DEVICE)
        if not os.path.exists(model_load_path):
             raise FileNotFoundError(f"Model file not found at {model_load_path}. Please train the model.")
        model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        model = None
    except Exception as e:
        print(f"ERROR: Failed to load model: {type(e).__name__} - {e}")
        model = None
else:
     print("ERROR: Model class not imported, cannot load model.")


# --- Ensure Upload Directory Exists ---
# Ensure UPLOAD_FOLDER path is relative to where uvicorn runs (phase_2)
upload_dir_path = os.path.join(os.getcwd(), UPLOAD_FOLDER) if os.path.basename(os.getcwd()) == 'phase_2' else UPLOAD_FOLDER
os.makedirs(upload_dir_path, exist_ok=True)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main upload page."""
    if not templates:
         return HTMLResponse("Server Configuration Error: Templates not loaded.", status_code=500)
    return templates.TemplateResponse("upload.html", {"request": request})

# --- Endpoint for ZIP file upload ---
@app.post("/predict_patient", response_class=HTMLResponse)
async def predict_patient(request: Request, file: UploadFile = File(...)):
    """Handles ZIP file upload containing DICOMs, runs inference, returns results."""
    if not model:
        return HTMLResponse("Server Error: Model not available.", status_code=503)
    if not templates:
         return HTMLResponse("Server Configuration Error: Templates not loaded.", status_code=500)
    if not full_inference_with_explanation:
         return HTMLResponse("Server Configuration Error: Inference function not available.", status_code=500)

    # Basic check for ZIP file type
    if not file.filename.lower().endswith(".zip"):
         return templates.TemplateResponse("result.html", {
             "request": request, "stage": "Error: Invalid file type. Please upload a .zip file.",
             "ct_image_url": None, "gradcam_image_url": None, "shap_image_url": None # Pass None for URLs
         }, status_code=400)

    # Define paths relative to phase_2 (where uvicorn runs)
    upload_dir = UPLOAD_FOLDER
    output_dir = OUTPUT_FOLDER

    # Generate a unique subfolder name for extraction within uploads/
    timestamp = f"{int(time.time())}_{np.random.randint(1000):03d}"
    base_filename = os.path.splitext(file.filename)[0]
    safe_base_filename = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in base_filename)
    extraction_folder = os.path.join(upload_dir, f"{safe_base_filename}_{timestamp}") # e.g., uploads/PatientX_12345
    zip_save_path = f"{extraction_folder}.zip"

    extraction_folder_abs = os.path.abspath(extraction_folder) # Use absolute path for processing
    output_dir_abs = os.path.abspath(output_dir)

    try:
        os.makedirs(upload_dir, exist_ok=True)
        print(f"Saving uploaded ZIP to: {zip_save_path}")
        # Save uploaded zip file temporarily
        with open(zip_save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract the zip file
        print(f"Extracting ZIP to: {extraction_folder_abs}")
        os.makedirs(extraction_folder_abs, exist_ok=True)
        with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
            # Check for nested structures or extract directly
            zip_ref.extractall(extraction_folder_abs)
        print("Extraction complete.")

    except zipfile.BadZipFile:
        print(f"Error: Uploaded file is not a valid ZIP archive: {file.filename}")
        # Clean up zip if extraction failed
        if os.path.exists(zip_save_path): os.remove(zip_save_path)
        if os.path.exists(extraction_folder_abs): shutil.rmtree(extraction_folder_abs)
        return templates.TemplateResponse("result.html", {
             "request": request, "stage": "Error: Invalid ZIP file.",
             "ct_image_url": None, "gradcam_image_url": None, "shap_image_url": None
         }, status_code=400)
    except Exception as e:
        print(f"Error handling upload/extraction: {type(e).__name__} - {e}")
        # Clean up
        if os.path.exists(zip_save_path): os.remove(zip_save_path)
        if os.path.exists(extraction_folder_abs): shutil.rmtree(extraction_folder_abs)
        return templates.TemplateResponse("result.html", {
             "request": request, "stage": f"Error processing upload: {e}",
             "ct_image_url": None, "gradcam_image_url": None, "shap_image_url": None
         }, status_code=500)
    finally:
        await file.close()
        # Clean up the temporary zip file NOW
        if os.path.exists(zip_save_path):
            try:
                print(f"Removing temporary ZIP file: {zip_save_path}")
                os.remove(zip_save_path)
            except Exception as e_clean:
                print(f"Error removing temp zip {zip_save_path}: {e_clean}")


    # --- Run Inference ---
    heatmap_url = None
    shap_url = None
    ct_url = None
    stage_result = "Error: Inference failed"
    pred_result = -1

    try:
        # Check if extraction folder actually contains files/subdirs (better check needed if nested)
        if not os.listdir(extraction_folder_abs):
             raise ValueError("Extracted folder is empty.")

        # Call the inference function with absolute paths for clarity
        pred_result, stage_result, ct_fname, gradcam_fname, shap_fname = \
            full_inference_with_explanation(
                model=model,
                dicom_folder=extraction_folder_abs,  # Pass the directory containing DICOMs
                output_folder=output_dir_abs  # Save images to absolute ./outputs/ path
            )

        # Convert filenames (relative to output_dir) to URL paths
        ct_url = f"/{OUTPUT_FOLDER}/{ct_fname}" if ct_fname else None
        heatmap_url = f"/{OUTPUT_FOLDER}/{gradcam_fname}" if gradcam_fname else None
        shap_url = f"/{OUTPUT_FOLDER}/{shap_fname}" if shap_fname else None

        print(f"Inference processing complete. Stage: {stage_result}")
        # --- Add Explanation Text Generation ---
        explanation_text = f"The model's prediction is '{stage_result}'. "
        if heatmap_url and shap_url:
            explanation_text += "Grad-CAM highlights the general input regions the model focused on. SHAP shows the specific contribution of each pixel (Red increases likelihood of prediction, Blue decreases)."
        elif heatmap_url:
            explanation_text += "Grad-CAM highlights the general input regions the model focused on for this prediction."
        elif shap_url:
            explanation_text += "SHAP shows the contribution of each pixel (Red increases likelihood of prediction, Blue decreases)."
        else:
            explanation_text += "No visual explanations were successfully generated."
    except Exception as e_infer:
        print(f"ERROR during inference call: {type(e_infer).__name__} - {e_infer}")
        traceback.print_exc()  # Log full traceback for server logs
        stage_result = f"Error during diagnosis: {e_infer}"

    finally:
        # --- Clean up the extracted folder ---
        try:
            if os.path.isdir(extraction_folder_abs):
                print(f"Removing extracted folder: {extraction_folder_abs}")
                shutil.rmtree(extraction_folder_abs)
        except Exception as e_clean_dir:
            print(f"Error removing extracted folder {extraction_folder_abs}: {e_clean_dir}")