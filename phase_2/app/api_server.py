import sys
import os
import uuid
from pathlib import Path
import shutil
import torch
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import PulmonaryInfarction3DCNN
from src.utils.dicom_loader import get_views
from src.inference import full_inference_with_explanation

# Constants
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)
MODEL_PATH = "models/trained_model.pth"  # Update with actual path

# Initialize FastAPI
app = FastAPI(
    title="Pulmonary Infarction Detection API",
    description="API for detecting pulmonary infarction from DICOM CT scans",
    version="0.1.0"
)

# Add static files mount
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Initialize templates
templates = Jinja2Templates(directory="app/templates")

# Task status tracking
task_status = {}

# Load model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = PulmonaryInfarction3DCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False
    model = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with upload form"""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "model_loaded": model_loaded
    })

@app.post("/upload_dicom")
async def upload_dicom(request: Request, background_tasks: BackgroundTasks, 
                       file: UploadFile = File(...)):
    """
    Upload a DICOM file and start processing in the background
    """
    # Generate unique ID for this task
    task_id = str(uuid.uuid4())
    
    # Create folder for this task
    task_folder = UPLOAD_FOLDER / task_id
    task_folder.mkdir(exist_ok=True)
    
    # Save file
    file_path = task_folder / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize task status
    task_status[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting processing..."
    }
    
    # Start background processing
    background_tasks.add_task(
        process_dicom_task, 
        task_id=task_id,
        file_path=str(file_path),
        model=model,
        device=device
    )
    
    # Return task_id for client to check status
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of a processing task"""
    if task_id not in task_status:
        return JSONResponse(
            status_code=404,
            content={"error": "Task not found"}
        )
    
    return task_status[task_id]

@app.get("/results/{task_id}", response_class=HTMLResponse)
async def get_results(request: Request, task_id: str):
    """View results of a completed task"""
    if task_id not in task_status or task_status[task_id]["status"] != "completed":
        return templates.TemplateResponse("processing.html", {
            "request": request,
            "task_id": task_id
        })
    
    result_data = task_status[task_id]["results"]
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": result_data["prediction"],
        "stage": result_data["stage"],
        "probability": result_data["probability"],
        "heatmap_path": result_data["heatmap_path"]
    })

async def process_dicom_task(task_id: str, file_path: str, model, device):
    """
    Process DICOM files in the background and update task status
    """
    try:
        # Update status
        task_status[task_id]["progress"] = 10
        task_status[task_id]["message"] = "Loading DICOM files..."
        
        # Get DICOM directory (assuming file_path is a DICOM file in a series)
        dicom_dir = os.path.dirname(file_path)
        
        # Update status
        task_status[task_id]["progress"] = 30
        task_status[task_id]["message"] = "Running model inference..."
        
        # Run inference
        predicted_class, class_probabilities, heatmap_path = full_inference_with_explanation(
            dicom_dir, 
            model_path=MODEL_PATH if model is None else None  # Only pass path if model not loaded
        )
        
        # Update status
        task_status[task_id]["progress"] = 90
        task_status[task_id]["message"] = "Finalizing results..."
        
        # Prepare results
        # Map predicted class to diagnosis (modify based on your model's classes)
        diagnosis_map = {
            0: "No Pulmonary Infarction",
            1: "Pulmonary Infarction"
        }
        
        # Map to stage (modify based on your staging logic)
        stage_map = {
            0: "N/A",
            1: "Stage I"  # Add more stages as needed
        }
        
        prediction = diagnosis_map.get(predicted_class, "Unknown")
        stage = stage_map.get(predicted_class, "Unknown")
        
        # Store results
        task_status[task_id]["results"] = {
            "prediction": prediction,
            "stage": stage,
            "probability": float(class_probabilities[predicted_class]),
            "heatmap_path": heatmap_path
        }
        
        # Mark as completed
        task_status[task_id]["status"] = "completed"
        task_status[task_id]["progress"] = 100
        task_status[task_id]["message"] = "Processing completed"
        
    except Exception as e:
        task_status[task_id]["status"] = "error"
        task_status[task_id]["message"] = f"Error: {str(e)}"
        print(f"Error processing task {task_id}: {e}")