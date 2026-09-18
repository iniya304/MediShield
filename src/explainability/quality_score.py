def calculate_quality_score(deletion_auc, insertion_auc, stability_score):
    """
    Aggregates explainability metrics into a single Quality Score (Q).
    Q = 0.5 * Faithfulness + 0.5 * Stability
    """
    faithfulness = ((1.0 - deletion_auc) + insertion_auc) / 2.0
    
    Q = (0.5 * faithfulness) + (0.5 * stability_score)
    return Q
