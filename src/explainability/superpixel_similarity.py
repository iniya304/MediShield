import numpy as np
from skimage.segmentation import slic

def calculate_superpixel_alignment(image_np, cam, n_segments=100):
    """
    Computes how well the Grad-CAM aligns with actual visual boundaries (superpixels)
    using SLIC (Simple Linear Iterative Clustering).
    """
    segments = slic(image_np, n_segments=n_segments, compactness=10, start_label=1)
    
    total_variance = 0
    for seg_id in np.unique(segments):
        mask = (segments == seg_id)
        cam_values = cam[mask]
        total_variance += np.var(cam_values) * np.sum(mask)
        
    total_variance /= (cam.shape[0] * cam.shape[1])
    alignment_score = max(0, 1.0 - (total_variance * 10))
    return alignment_score
