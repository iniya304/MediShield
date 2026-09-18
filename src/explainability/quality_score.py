def calculate_quality_score(deletion_auc, insertion_auc, robustness_score, consistency_score):
    """
    Aggregates explainability metrics into a single Quality Score (Q).
    Faithfulness uses Quantus (Deletion/Insertion).
    Stability uses custom SSIM (Robustness/Consistency).
    """
    faithfulness = ((1.0 - deletion_auc) + insertion_auc) / 2.0
    stability = (robustness_score + consistency_score) / 2.0
    
    Q = (0.5 * faithfulness) + (0.5 * stability)
    return Q
