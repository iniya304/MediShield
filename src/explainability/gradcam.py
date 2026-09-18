from pytorch_grad_cam import GradCAM as BaseGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class GradCAM:
    def __init__(self, model, target_layers):
        """
        Wrapper around pytorch-grad-cam library.
        target_layers should be a list of layers, e.g. [model.features[-1]]
        """
        self.model = model
        self.cam = BaseGradCAM(model=model, target_layers=target_layers)
        
    def generate(self, input_tensor, target_class=None):
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            targets = None # Automatically targets the highest scoring class
            
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        # Returns [batch_size, H, W], we take the first item
        return grayscale_cam[0, :]
