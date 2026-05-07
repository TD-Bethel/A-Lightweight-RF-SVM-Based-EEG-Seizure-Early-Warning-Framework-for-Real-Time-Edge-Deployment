"""
neurowatch_selector.py  --  Shared RF importance feature selector.

Kept in its own module so joblib/pickle can resolve the class
from any script (train, inference, Pi engine) without __main__ conflicts.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RFImportanceSelector:
    """Select top-k features ranked by a bootstrap Random Forest importance."""

    def __init__(self, k=75):
        self.k = k
        self.top_indices_ = None
        self.importances_ = None

    def fit(self, X, y, n_estimators=300, max_depth=10, random_state=42):
        rf_boot = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        rf_boot.fit(X, y)
        self.importances_ = rf_boot.feature_importances_
        self.top_indices_ = np.sort(
            np.argsort(self.importances_)[::-1][:self.k]
        )
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.top_indices_]

    def fit_transform(self, X, y, **kw):
        return self.fit(X, y, **kw).transform(X)
