import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve


class ImbalancedLogisticRegression(LogisticRegression):

    def fit(self, X, y, sample_weight=None):
        self.train_x = X
        self.train_y = y
        super(ImbalancedLogisticRegression, self).fit(X, y, sample_weight)
        return self

    def predict(self, X):
        precision, recall, thresholds_pr = precision_recall_curve(self.train_y, self.predict_proba(self.train_x)[:, 1])
        start = int(len(precision) / 3)
        best_idx = np.argmax((2 * precision[start:] * recall[start:]) / (precision[start:] + recall[start:]))
        self.best_threshold = thresholds_pr[best_idx]
        y_pred_custom = (self.predict_proba(X)[:, 1] > self.best_threshold).astype(int)
        return y_pred_custom

    def score(self, X, y, labels=None, sample_weight=None, normalize=None):
        "Calculate the score according to the threshold of f-score"
        print("I'm a CustomLogisticRegression score, if you see me many times probably I called by GridSearchCV")
        y_pred = self.predict(X)
        return (y_pred == y).sum() / len(y)

