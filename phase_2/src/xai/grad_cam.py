import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from src.utils.image_preprocessing import load_dicom_image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])  # (C)
        activations = self.activations.squeeze(0)  # (C, D, H, W)

        for i in range(activations.shape[0]):
            activations[i, :, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

# Function to Generate and Save GRAD-CAM Heatmap
def generate_grad_cam(model, dicom_folder, predicted_class, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model.to(device)
    
    # Preprocess
    scan_tensor = load_dicom_image(dicom_folder)
    scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    # Find the target convolutional layer
    target_layer = model.features[-2]  # Last conv layer before FC layers (adjust based on cnn_model.py!)
    
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(scan_tensor, class_idx=predicted_class)
    
    grad_cam.remove_hooks()

    # Save heatmap as image
    os.makedirs('outputs', exist_ok=True)
    heatmap_path = os.path.join('outputs', 'grad_cam_heatmap.png')

    plt.figure(figsize=(8,8))
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.title('GRAD-CAM for Pulmonary Infarction')
    plt.savefig(heatmap_path)
    plt.close()

    return heatmap_path
