import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# Step 1: Create necessary directories
os.makedirs('src/models/saved_models', exist_ok=True)
save_path = os.path.join('src', 'models', 'saved_models', 'best_model.pth')

# Step 2: Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Step 3: Load dummy dataset (replace this with your real data loader)
train_dataset = datasets.FakeData(transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

# Step 4: Define model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load checkpoint if exists
if os.path.exists(save_path):
    print("Loading existing model weights...")
    model.load_state_dict(torch.load(save_path))

# Step 5: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training loop
epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Step 7: Save the model
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
