import torch
import torchvision.transforms.functional as F
import random

def add_gaussian_noise(tensor, std=0.1):
    noise = torch.randn_like(tensor) * std
    return tensor + noise

def apply_blur(tensor, kernel_size=5):
    # kernel_size must be odd
    return F.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size])

def apply_brightness(tensor, brightness_factor=1.5):
    return F.adjust_brightness(tensor, brightness_factor)

class StressTestTransforms:
    """
    Randomly applies one of the stress conditions (noise, blur, or brightness) 
    at a specified severity level to evaluate model robustness.
    """
    def __init__(self, severity=1):
        self.severity = severity
        
    def __call__(self, img):
        choice = random.choice(['noise', 'blur', 'brightness'])
        
        if choice == 'noise':
            return add_gaussian_noise(img, std=0.05 * self.severity)
        
        elif choice == 'blur':
            # Severity 1 -> 3, Severity 2 -> 5, Severity 3 -> 7
            kernel = 1 + (self.severity * 2) 
            return apply_blur(img, kernel_size=kernel)
            
        elif choice == 'brightness':
            factor = 1.0 + (0.2 * self.severity)
            return apply_brightness(img, brightness_factor=factor)
            
        return img
