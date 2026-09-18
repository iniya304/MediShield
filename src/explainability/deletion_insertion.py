import quantus

def calculate_deletion_insertion_auc(model, image, cam, target_class):
    """
    Computes Deletion and Insertion Area Under Curve (AUC) using the Quantus XAI library.
    image: [1, 3, H, W] tensor
    cam: [H, W] numpy array
    target_class: int
    """
    a_batch = cam.reshape(1, *cam.shape)
    x_batch = image.cpu().numpy()
    y_batch = [target_class]
    
    # Initialize metrics
    deletion_metric = quantus.Deletion(
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        disable_warnings=True
    )
    
    insertion_metric = quantus.Insertion(
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        disable_warnings=True
    )
    
    # Calculate scores (returns list of scores per batch item)
    del_scores = deletion_metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch)
    ins_scores = insertion_metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch)
    
    return del_scores[0], ins_scores[0]
