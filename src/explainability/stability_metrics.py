import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_cam_stability(cam_original, cam_perturbed):
    """
    Evaluates how stable the explanation is under minor image perturbations using SSIM.
    Higher SSIM means the explanation is more robust.
    """
    score, _ = ssim(cam_original, cam_perturbed, full=True, data_range=1.0)
    return score
