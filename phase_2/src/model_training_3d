import torch.nn as nn  #torch nn for neural netwrk layers and loss layers 
import torch.optim as optim #optimizers
import torch
from monai.networks.nets import DenseNet121

class PulmonaryModel(nn.Module):
    def __init__(self):
        super(PulmonaryModel, self).__init__()
        self.model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)  # Binary Classification

    def forward(self, x):
        return self.model(x)

model = PulmonaryModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#(Adaptive Moment Estimation) to update model weights.