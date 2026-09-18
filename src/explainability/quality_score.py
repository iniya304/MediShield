def calculate_quality_score(deletion_auc, insertion_auc, stability_score, superpixel_score):
    """
    Aggregates explainability metrics into a single Quality Score (Q).
    Q = 0.4 * Faithfulness + 0.3 * Stability + 0.3 * Alignment
    """
    faithfulness = ((1.0 - deletion_auc) + insertion_auc) / 2.0
    
    Q = (0.4 * faithfulness) + (0.3 * stability_score) + (0.3 * superpixel_score)
    return Q
