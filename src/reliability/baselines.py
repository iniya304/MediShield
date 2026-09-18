import numpy as np
from scipy.stats import entropy

def msp_baseline(probs):
    """
    Maximum Softmax Probability (MSP).
    A standard baseline for confidence/reliability.
    """
    return np.max(probs, axis=1)

def entropy_baseline(probs):
    """
    Negative entropy as a reliability baseline (higher means more reliable).
    """
    return -entropy(probs.T)
