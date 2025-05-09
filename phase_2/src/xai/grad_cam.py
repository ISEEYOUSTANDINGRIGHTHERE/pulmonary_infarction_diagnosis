import torch
import numpy as np
import torch.nn.functional as F # Make sure F is imported

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        if self.target_layer is None:
             raise ValueError("Target layer cannot be None for GradCAM")

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
             self.activations = output.detach()
        elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
             self.activations = output[0].detach()
        else: self.activations = None

    def _backward_hook(self, module, grad_input, grad_output):
        if grad_output and isinstance(grad_output[0], torch.Tensor):
            self.gradients = grad_output[0].detach()
        else: self.gradients = None

    def remove_hooks(self):
        if hasattr(self, 'forward_handle') and self.forward_handle: self.forward_handle.remove()
        if hasattr(self, 'backward_handle') and self.backward_handle: self.backward_handle.remove()
        self.gradients = None
        self.activations = None

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None: target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
             print("ERROR: Gradients or Activations not captured by hooks.")
             # Return a dummy shape consistent with previous logs if possible
             # Get expected H, W from input tensor spatial dims if activations failed early- imp to notice
             h, w = input_tensor.shape[-2:]
             return np.zeros((h // 4, w // 4)) # Example dummy shape for explanation

        # --- **** MODIFICATION HERE **** --- as asked by kumar sir
        # Calculate weights using ReLU on gradients first (positive influence)
        positive_gradients = F.relu(self.gradients)
        pooled_gradients = torch.mean(positive_gradients, dim=[0, 2, 3, 4])  # Shape: [C]
        # --- *************************** ---

        activations = self.activations.squeeze(0)  # Shape: [C, D, H, W]

        # Check dimensions before weighting
        if pooled_gradients.shape[0] != activations.shape[0]:
             print(f"ERROR: Mismatch between gradient channels ({pooled_gradients.shape[0]}) and activation channels ({activations.shape[0]})")
             h, w = activations.shape[-2:]
             return np.zeros((h, w)) # Return empty map of activation size

        # Weight activations (in-place)
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]

        # Apply ReLU BEFORE combining channels
        activations = F.relu(activations)

        # Combine channels using SUM
        heatmap_3d = torch.sum(activations, dim=0) # Shape: [D, H, W]
        heatmap_np = heatmap_3d.cpu().numpy()

        # Normalize 0-1 range
        h_max = np.max(heatmap_np)
        h_min = np.min(heatmap_np)
        if h_max - h_min > 1e-8:
             heatmap_norm = (heatmap_np - h_min) / (h_max - h_min)
        else:
             print("Warning: Grad-CAM heatmap is all zeros/constant after modifications.")
             heatmap_norm = np.zeros_like(heatmap_np)

        # Select middle slice
        if heatmap_norm.ndim == 3 and heatmap_norm.shape[0] > 0:
            mid_slice_idx = heatmap_norm.shape[0] // 2
            final_heatmap_slice = heatmap_norm[mid_slice_idx]
        elif heatmap_norm.ndim == 2:
             final_heatmap_slice = heatmap_norm
        else:
             print(f"Warning: Unexpected heatmap dimension: {heatmap_norm.ndim}. Returning zeros.")
             h, w = self.activations.shape[-2:]
             final_heatmap_slice = np.zeros((h, w))

        return final_heatmap_slice