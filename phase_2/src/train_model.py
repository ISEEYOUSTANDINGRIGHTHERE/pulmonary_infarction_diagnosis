import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import YourModel  # Replace with actual class name
from dataset.your_dataset import YourDataset  # Replace with actual dataset
import os

# 1. Load your dataset
train_dataset = YourDataset(...)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 2. Initialize model, loss, optimizer
model = YourModel()
criterion = nn.CrossEntropyLoss()  # or your loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
epochs = 10
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs('C:\Users\Akash\OneDrive\Documents\GitHub\pulmonary_infarction_diagnosis\phase_2\src\models\saved_models', exist_ok=True)
        torch.save(model.state_dict(), 'C:\Users\Akash\OneDrive\Documents\GitHub\pulmonary_infarction_diagnosis\phase_2\src\models\saved_models/best_model.pth')
        print(" Saved new best model")

