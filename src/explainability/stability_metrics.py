import quantus

def calculate_cam_stability(model, image, cam, target_class):
    """
    Evaluates how stable the explanation is under minor perturbations using Quantus (Relative Input Stability).
    Higher stability means the explanation is more robust.
    """
    a_batch = cam.reshape(1, *cam.shape)
    x_batch = image.cpu().numpy()
    y_batch = [target_class]
    
    # Relative Input Stability (RIS) evaluates robustness to noise
    ris_metric = quantus.RelativeInputStability(
        nr_samples=10,
        disable_warnings=True
    )
    
    ris_scores = ris_metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch)
    
    return ris_scores[0]
