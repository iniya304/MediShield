import xgboost as xgb

class XGBoostReliabilityDetector:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='logloss'
        )
        
    def train(self, X_train, y_train):
        """
        X_train: Feature matrix (e.g., Margin, Entropy, Perturbation-Delta, Deep Features)
        y_train: Binary labels (1 = Correct prediction, 0 = Incorrect prediction)
        """
        self.model.fit(X_train, y_train)
        
    def predict_reliability(self, X):
        """
        Returns probability of the base model being correct.
        """
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, filepath):
        self.model.save_model(filepath)
        
    def load(self, filepath):
        self.model.load_model(filepath)
