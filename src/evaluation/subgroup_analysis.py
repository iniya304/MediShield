import pandas as pd

def analyze_subgroups(df, attribute, metric_fn):
    """
    Evaluates a specific metric across different subgroups defined by `attribute`.
    """
    results = {}
    for group in df[attribute].unique():
        subgroup_data = df[df[attribute] == group]
        if len(subgroup_data) > 0:
            results[group] = metric_fn(subgroup_data)
    return results
