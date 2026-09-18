import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as F
import random

def compute_ssim(cam1, cam2):
    # Ensure they are proper range [0, 1]
    return ssim(cam1, cam2, data_range=1.0)

def calculate_robustness_score(model, image, original_cam, target_class, gradcam_generator, num_perturbations=5):
    """
    R = 1/|T| sum(SSIM(S, S_t))
    Applies minor noise perturbations and measures SSIM stability.
    """
    scores = []
    
    for _ in range(num_perturbations):
        # Apply a random perturbation (Gaussian noise)
        std = random.uniform(0.01, 0.05)
        noise = torch.randn_like(image) * std
        perturbed_image = torch.clamp(image + noise, 0, 1)
        
        # Generate new CAM
        perturbed_cam = gradcam_generator.generate(perturbed_image, target_class)
        
        # Compare
        scores.append(compute_ssim(original_cam, perturbed_cam))
        
    return np.mean(scores) if scores else 0.0

def calculate_consistency_score(model, image, target_class, gradcam_generator, num_augmentations=4):
    """
    C = 2/(M(M-1)) sum_{i<j} SSIM(S_i, S_j)
    Applies benign augmentations (horizontal flip) and measures pairwise SSIM stability.
    """
    cams = []
    
    for i in range(num_augmentations):
        # Apply augmentation (just flips for structural simplicity)
        aug_img = image.clone()
        if i % 2 == 1:
            aug_img = F.hflip(aug_img)
            
        cam = gradcam_generator.generate(aug_img, target_class)
        
        # Reverse the geometric transformation to align it for comparison
        if i % 2 == 1:
            cam = np.fliplr(cam)
            
        cams.append(cam)
        
    # Pairwise SSIM
    M = len(cams)
    if M < 2: return 0.0
    
    total_ssim = 0
    for i in range(M):
        for j in range(i+1, M):
            total_ssim += compute_ssim(cams[i], cams[j])
            
    return (2.0 / (M * (M - 1))) * total_ssim
