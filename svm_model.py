import numpy as np

class SVMClassifier:
    """Wraps a trained SVC and gives a 0-1 attack score from its decision function."""

    def __init__(self, svc):
        self.svc = svc

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.svc.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.svc.predict(X)