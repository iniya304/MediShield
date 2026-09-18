import torch
import numpy as np
from sklearn.metrics import auc

def calculate_deletion_insertion_auc(model, image, cam, num_steps=100):
    """
    Computes Deletion and Insertion Area Under Curve (AUC).
    image: [1, 3, H, W]
    cam: [H, W] normalized heatmap
    """
    H, W = cam.shape
    total_pixels = H * W
    step_size = total_pixels // num_steps
    
    # Sort pixels by saliency (highest to lowest)
    flat_cam = cam.flatten()
    sorted_indices = np.argsort(flat_cam)[::-1]
    
    del_scores = []
    ins_scores = []
    
    del_img = image.clone()
    ins_img = torch.zeros_like(image)
    
    model.eval()
    with torch.no_grad():
        original_pred = model(image).softmax(dim=1)
        target_class = original_pred.argmax(dim=1).item()
        
    for i in range(num_steps):
        idx_to_modify = sorted_indices[i * step_size : (i+1) * step_size]
        
        # Deletion: mask out most salient pixels
        del_img_flat = del_img.view(3, -1)
        del_img_flat[:, idx_to_modify] = 0
        del_img = del_img_flat.view(1, 3, H, W)
        
        # Insertion: add most salient pixels from original image to blank image
        ins_img_flat = ins_img.view(3, -1)
        orig_img_flat = image.view(3, -1)
        ins_img_flat[:, idx_to_modify] = orig_img_flat[:, idx_to_modify]
        ins_img = ins_img_flat.view(1, 3, H, W)
        
        with torch.no_grad():
            del_pred = model(del_img).softmax(dim=1)[0, target_class].item()
            ins_pred = model(ins_img).softmax(dim=1)[0, target_class].item()
            
        del_scores.append(del_pred)
        ins_scores.append(ins_pred)
        
    del_auc = auc(np.linspace(0, 1, num_steps), del_scores)
    ins_auc = auc(np.linspace(0, 1, num_steps), ins_scores)
    
    return del_auc, ins_auc
