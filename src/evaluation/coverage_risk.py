import numpy as np

def calculate_coverage_risk(confidences, correctness, thresholds):
    """
    Computes coverage and risk at varying confidence thresholds.
    """
    coverage = []
    risk = []
    
    for t in thresholds:
        accepted_mask = confidences >= t
        cov = np.mean(accepted_mask)
        if cov > 0:
            rsk = 1.0 - np.mean(correctness[accepted_mask])
        else:
            rsk = 0.0
            
        coverage.append(cov)
        risk.append(rsk)
        
    return np.array(coverage), np.array(risk)
