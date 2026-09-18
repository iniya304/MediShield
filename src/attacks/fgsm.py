import torch

def fgsm_attack(image, epsilon, data_grad):
    """
    Applies Fast Gradient Sign Method (FGSM) to the image.
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    
    return perturbed_image
