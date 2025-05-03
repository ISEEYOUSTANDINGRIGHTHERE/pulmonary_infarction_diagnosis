import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# Assuming these utility functions and model class are correctly defined in your project structure
from src.utils.dicom_loader import load_dicom_volume, get_views
# Make sure this path matches your actual model file location
from models.cnn_model import PulmonaryInfarction3DCNN

# --- Config ---
# !!! IMPORTANT: Verify this is the correct path to the directory CONTAINING PAT001, PAT002 etc. folders !!!
DATA_DIR = "D:\\PulmonaryInfarction\\Phase2dataset"
# Path to the CSV file within your project structure
LABEL_CSV = "metadata/patient_labels.csv" # Relative path from phase_2 directory
LABEL_MAP = {"NoPI": 0, "Stage1": 1, "Stage2": 2, "Stage3": 3}
# !!! IMPORTANT: Corrected path based on screenshot !!!
SAVE_PATH = "models/pulmonary_model.pth" # Relative path from phase_2 directory

BATCH_SIZE = 2
EPOCHS = 2 # Keep low for debugging, increase for actual training
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Dataset ---
class PILDataset(Dataset):
    def __init__(self, patient_ids, label_dict, data_dir):
        self.patient_ids = patient_ids
        self.label_dict = label_dict
        self.data_dir = data_dir # Store data_dir

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        # --- Debug Print: Show which ID is being loaded ---
        print(f"Attempting to load patient ID: {patient_id!r}") # Use !r for clear representation
        # ----------------------------------------------------
        dicom_path = os.path.join(self.data_dir, patient_id) # Use self.data_dir

        # --- Debug Check: Verify path exists before loading ---
        # Optional, but can give immediate feedback if path is wrong
        if not os.path.isdir(dicom_path):
            print(f"ERROR in __getitem__: Directory not found: {dicom_path}")
            # Raise error to stop DataLoader, helps identify issues faster than waiting for os.listdir
            raise FileNotFoundError(f"Constructed path not found in __getitem__: {dicom_path}")
        # -------------------------------------------------------

        # Load volume - This is where the FileNotFoundError or ValueError might occur if path is wrong or folder is empty/lacks .dcm
        try:
            volume = load_dicom_volume(dicom_path)  # [D, H, W]
        except FileNotFoundError:
             # Should be caught by the check above, but belt-and-suspenders
             print(f"ERROR: load_dicom_volume failed (FileNotFoundError) for path: {dicom_path}")
             raise
        except ValueError as e:
             # Catches "need at least one array to stack" or other load_dicom_volume issues
             print(f"ERROR: load_dicom_volume failed (ValueError: {e}) for path: {dicom_path}")
             raise
        except Exception as e:
             # Catch any other unexpected errors during loading
             print(f"ERROR: load_dicom_volume failed (Unexpected error: {e}) for path: {dicom_path}")
             raise


        # Process volume
        axial, _, _ = get_views(volume)       # Choose axial view, assumes get_views handles resizing etc.
        # Add channel dimension: [1, D, H, W]
        volume_tensor = torch.tensor(axial, dtype=torch.float32).unsqueeze(0)

        # Get label
        try:
             label_str = self.label_dict[patient_id]
             label = LABEL_MAP[label_str]
        except KeyError:
             print(f"ERROR: patient_id '{patient_id}' not found in label_dict or label '{label_str}' not in LABEL_MAP.")
             # Handle missing labels appropriately - here we raise an error
             raise KeyError(f"Label lookup failed for patient_id: {patient_id}")

        return volume_tensor, torch.tensor(label, dtype=torch.long)

# --- Load CSV ---
print(f"Loading labels from: {LABEL_CSV}")
try:
    df = pd.read_csv(LABEL_CSV)
    # Convert patient_id column to string just in case they are read as numbers
    df['patient_id'] = df['patient_id'].astype(str)
    label_dict = dict(zip(df.patient_id, df.label))
    all_patients = list(label_dict.keys())
    print(f"Loaded {len(all_patients)} patient entries from CSV.")
except FileNotFoundError:
    print(f"ERROR: Label CSV file not found at {LABEL_CSV}")
    exit() # Exit if CSV is missing
except KeyError as e:
    print(f"ERROR: Missing expected column '{e}' in {LABEL_CSV}")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load or process {LABEL_CSV}: {e}")
    exit()


# --- Train/Val Split ---
print("Splitting data into training and validation sets...")
train_ids, val_ids = train_test_split(all_patients, test_size=0.2, random_state=42) # Use fixed random state for reproducibility
print(f"Training set size: {len(train_ids)}")
print(f"Validation set size: {len(val_ids)}")


# --- Pre-Loop Debug Checks ---
print("-" * 20)
print("--- Pre-Loop Checks ---")
print(f"DATA_DIR used: {DATA_DIR}")
print(f"LABEL_CSV used: {LABEL_CSV}")
print(f"Total patients from CSV: {len(all_patients)}")
print(f"First 5 patient IDs from CSV: {all_patients[:5]}")
print(f"Total training IDs after split: {len(train_ids)}")
print(f"First 5 training IDs: {train_ids[:5]}")
print(f"Total validation IDs after split: {len(val_ids)}")
print(f"First 5 validation IDs: {val_ids[:5]}")
print("-" * 20)

# Explicitly check the very first training ID before the loop starts
if train_ids:
    first_train_id = train_ids[0]
    print(f"Checking first training ID: {first_train_id!r}")
    first_path_check = os.path.join(DATA_DIR, first_train_id)
    print(f"Path constructed for first ID: {first_path_check}")
    if os.path.isdir(first_path_check):
         print("-> Path for first training ID EXISTS.")
    else:
         print("!!! ERROR -> Path for the very first training ID DOES NOT EXIST.")
         print("!!! Please check the patient ID, DATA_DIR, and the contents of the DATA_DIR. !!!")
         # Consider exiting if the first ID is already problematic
         # exit()
else:
    print("!!! ERROR: train_ids list is empty after split. Check CSV or split logic. !!!")
    exit() # Exit if no training data
print("-" * 20)
# --- End Pre-Loop Debug Checks ---


# --- Datasets and DataLoaders ---
# Pass DATA_DIR to the dataset instance
train_dataset = PILDataset(train_ids, label_dict, DATA_DIR)
val_dataset = PILDataset(val_ids, label_dict, DATA_DIR)

# Set num_workers=0 for easier debugging (avoids multiprocessing issues)
# Increase later for performance if needed and stable
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0) # Usually batch_size=1 and no shuffle for validation

# --- Model ---
print("Initializing model...")
# Ensure your model definition matches what was saved if loading weights
model = PulmonaryInfarction3DCNN(num_classes=len(LABEL_MAP)).to(DEVICE) # Pass number of classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
print("Model, criterion, optimizer initialized.")

# --- Training Loop ---
print("Starting training loop...")
for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    total_loss = 0
    # Use tqdm for progress bar
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for i, (x, y) in enumerate(train_iterator):
        try:
            x, y = x.to(DEVICE), y.to(DEVICE) # Move data to device

            optimizer.zero_grad() # Zero gradients
            out = model(x)        # Forward pass
            loss = criterion(out, y) # Calculate loss
            loss.backward()       # Backward pass
            optimizer.step()        # Update weights

            total_loss += loss.item()

            # Update tqdm description with current average loss
            train_iterator.set_postfix(loss=f"{total_loss / (i+1):.4f}")

        except Exception as e:
            print(f"\n!!! ERROR during training loop (batch {i}) !!!")
            print(f"Error type: {type(e)}")
            print(f"Error details: {e}")
            # Depending on the error, you might want to investigate the specific batch data
            # Or potentially skip the batch, but it's usually better to fix the root cause
            raise # Re-raise the error to stop training

    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    total_val_loss = 0
    val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    with torch.no_grad(): # Disable gradient calculations for validation
        for i, (x, y) in enumerate(val_iterator):
            try:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y) # Optionally calculate validation loss
                total_val_loss += loss.item()

                pred = out.argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()

                # Update tqdm description with current average loss & accuracy
                val_iterator.set_postfix(loss=f"{total_val_loss / (i+1):.4f}", acc=f"{100 * correct / total:.2f}%")

            except Exception as e:
                print(f"\n!!! ERROR during validation loop (batch {i}) !!!")
                print(f"Error type: {type(e)}")
                print(f"Error details: {e}")
                raise # Re-raise the error

    avg_val_loss = total_val_loss / len(val_loader)
    acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {acc:.4f} ({correct}/{total})")


# --- Save ---
print("Training finished. Saving model...")
try:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    # Save the model state dictionary
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved successfully to: {SAVE_PATH}")
except Exception as e:
    print(f"ERROR: Failed to save model to {SAVE_PATH}: {e}")

print("Script finished.")