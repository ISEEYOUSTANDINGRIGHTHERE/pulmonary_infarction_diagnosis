import os
import shutil
import pandas as pd

# Adjust these paths as needed
CSV_PATH = r"D:\PulmonaryInfarction\Phase2dataset\patient_labels.csv"
SOURCE_DIR = r"D:\PulmonaryInfarction\Phase2dataset"
TARGET_DIR = r"D:\PulmonaryInfarction\Phase2dataset_labeled"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Create label folders if not exist
label_set = df['label'].unique()
for label in label_set:
    os.makedirs(os.path.join(TARGET_DIR, label), exist_ok=True)

# Move patient folders
for _, row in df.iterrows():
    patient_id = row['patient_id']
    label = row['label']

    src = os.path.join(SOURCE_DIR, patient_id)
    dst = os.path.join(TARGET_DIR, label, patient_id)

    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {patient_id} → {label}")
    else:
        print(f" Missing: {src}")
