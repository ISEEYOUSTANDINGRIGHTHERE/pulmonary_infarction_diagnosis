import torch
import torch.nn as nn
import torch.nn.functional as F

class PulmonaryInfarction3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(PulmonaryInfarction3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 8 * 16 * 16, 512)  # depends on our final spatial dims but for now assume 128*8*16*16
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input x shape: (batch_size, 1, D, H, W)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

#  Quick test
if __name__ == "__main__":
    model = PulmonaryInfarction3DCNN()
    x = torch.randn((1, 1, 64, 128, 128))  # Dummy CT input
    output = model(x)
    print("Output shape:", output.shape)  # Should be [1, 4]
    
