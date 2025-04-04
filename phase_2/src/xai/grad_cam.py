import torch 
import numpy as np
import cv2
from grad_cam import GradCAM

def apply_gradcam(model,image_tensor, target layer):
    gradcam= GradCAM(model,target layer)
    heatmap=gradcam(image_tensor)
    return cv2.applyColorMap(np.uint8(255*heatmap),cv2.COLORMAP_JET)