class AbstentionLayer:
    def __init__(self, reliability_threshold=0.8, quality_threshold=0.7):
        self.T_R = reliability_threshold
        self.T_Q = quality_threshold
        
    def evaluate(self, reliability_prob, quality_score):
        """
        Dual-gate abstention logic.
        Returns:
            "ACCEPT" if both conditions are met.
            "REJECT_LOW_RELIABILITY" if reliability is too low.
            "REJECT_LOW_EXPLAINABILITY" if explainability quality is too low.
        """
        if reliability_prob < self.T_R:
            return "REJECT_LOW_RELIABILITY"
        if quality_score < self.T_Q:
            return "REJECT_LOW_EXPLAINABILITY"
            
        return "ACCEPT"
