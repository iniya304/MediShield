import numpy as np
from scipy.stats import entropy

def compute_margin(probs):
    """
    Computes the margin: Difference between the top and second-top probabilities.
    """
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]

def compute_entropy(probs):
    """
    Computes the Shannon entropy of the predicted probability distribution.
    """
    return entropy(probs.T)

def compute_perturbation_delta(original_probs, perturbed_probs):
    """
    Computes how much the model's confidence changes under a perturbation.
    """
    orig_conf = np.max(original_probs, axis=1)
    pert_conf = np.max(perturbed_probs, axis=1)
    return orig_conf - pert_conf
