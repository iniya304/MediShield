import numpy as np

def compute_bootstrap_ci(data, metric_fn, n_bootstraps=1000, alpha=0.05):
    """
    Computes Bootstrap Confidence Interval for a given metric.
    """
    n = len(data)
    bootstrapped_metrics = []
    
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, n)
        sample = data[indices]
        bootstrapped_metrics.append(metric_fn(sample))
        
    lower_bound = np.percentile(bootstrapped_metrics, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrapped_metrics, 100 * (1 - alpha / 2))
    mean_val = np.mean(bootstrapped_metrics)
    
    return mean_val, lower_bound, upper_bound
