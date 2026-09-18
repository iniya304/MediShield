import torch
import torch.nn as nn

def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=40):
    """
    Projected Gradient Descent (PGD) Attack.
    """
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()

    original_images = images.data.clone()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        
        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        # Update adversarial images
        adv_images = images + alpha * images.grad.sign()
        
        # Clip perturbations
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = original_images + eta
        
        # We omit the global [0,1] clamp here assuming inputs are already normalized via ImageNet mean/std
        images = images.detach_()

    return images
