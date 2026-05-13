"""
neurowatch_selector.py  --  Shared RF importance feature selector.

Kept in its own module so joblib/pickle can resolve the class
from any script (train, inference, Pi engine) without __main__ conflicts.
"""

# === SECTION 1 — WHY THIS FILE EXISTS =========================================
#
# When you save a Python object with joblib (e.g. joblib.dump(selector, "sel.pkl")),
# Python needs to be able to FIND the class definition again when loading it back
# (joblib.load). If you defined the class inside a notebook or a __main__ block,
# that fails because "__main__" doesn't exist in another script.
#
# Solution: define the class here in its own named module ("neurowatch_selector"),
# so that ANY script in the project can import it and joblib can resolve it.
#
# Every script that loads the saved selector must have:
#   from neurowatch_selector import RFImportanceSelector
# before calling joblib.load().
#
# === HOW IT FITS INTO THE PIPELINE ============================================
#
#  Training time (train_*.py):
#    1. Create RFImportanceSelector(k=75)
#    2. Call .fit(X_train, y_train)         → trains a "scout" RF to rank features
#    3. Call .transform(X_train)            → keeps only the top 75 columns
#    4. Save: joblib.dump(selector, "selector.pkl")
#    5. Train the real model on the reduced X
#
#  Inference time (Pi engine / dashboard):
#    1. Load: selector = joblib.load("selector.pkl")
#    2. Call .transform(X_new)              → slims incoming data to same 75 cols
#    3. Feed into the loaded model for prediction
#
# ==============================================================================

# === SECTION 2 — IMPORTS ======================================================

import numpy as np
# numpy is needed for two things:
#   np.asarray()  — converts any array-like (list, pandas DataFrame) to a numpy array
#   np.argsort()  — returns the indices that would sort an array (used to rank features)
#   np.sort()     — sorts the chosen indices so column ORDER is preserved after selection

from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier is the "scout" model used ONLY inside .fit() to score
# feature importance. It is not the main classifier saved by the project —
# that lives in the models/ directory. This RF exists just to rank features.


# === SECTION 3 — THE SELECTOR CLASS ===========================================

class RFImportanceSelector:
    """Select top-k features ranked by a bootstrap Random Forest importance.

    This is a lightweight sklearn-compatible transformer (fit / transform /
    fit_transform).  It is NOT a full sklearn Pipeline step (it doesn't inherit
    BaseEstimator), but it follows the same interface so it's easy to swap in
    later if needed.
    """

    # --- 3a. Constructor ------------------------------------------------------
    def __init__(self, k=75):
        """
        Set up the selector before any data is seen.

        Parameters
        ----------
        k : int
            How many features to keep.
            Default is 75, which was tuned to give a good accuracy/speed tradeoff
            on the NeuroWatch feature set (which typically has 200-300 columns).

        HOW TO CHANGE:
            - Increase k (e.g. k=100) if validation accuracy seems low — you may
              be dropping useful features.
            - Decrease k (e.g. k=50) if inference is too slow on the Pi, or if
              you have a smaller feature set.
            - k must be <= total number of features in your training data;
              otherwise np.argsort slicing will silently return all features.

        Attributes set here (all None until .fit() is called):
            self.top_indices_  — numpy array of column indices to keep, sorted
                                 in ascending order so the slice preserves the
                                 original feature ordering.
            self.importances_  — full importance scores for ALL features from
                                 the scout RF; useful for debugging / plotting.
        """
        self.k = k
        self.top_indices_ = None   # filled in by .fit(); None means "not fitted yet"
        self.importances_ = None   # filled in by .fit(); one float per input feature

    # --- 3b. Fit — rank features using a "scout" Random Forest ---------------
    def fit(self, X, y, n_estimators=300, max_depth=10, random_state=42):
        """
        Train a temporary Random Forest on (X, y) purely to rank feature importance.
        The RF itself is discarded after ranking — only the top-k column indices
        are stored for use during transform().

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Full training feature matrix (BEFORE any selection).
        y : array-like, shape (n_samples,)
            Class labels (0 = normal, 1 = seizure, etc.).
        n_estimators : int
            Number of trees in the scout RF.
            More trees → more stable importance scores, but slower to fit.
            300 is a good default; reduce to 100 for quick experiments.
        max_depth : int
            Maximum tree depth in the scout RF.
            Shallow trees (e.g. 5-10) generalise better and are less likely
            to rank noise features highly.
        random_state : int
            Random seed for reproducibility. Keep at 42 so that re-running
            training produces the same selected features.

        Returns
        -------
        self
            Returning self is the sklearn convention; it allows method chaining:
            selector.fit(X, y).transform(X)

        HOW TO CHANGE:
            - Change n_estimators to 100 for faster (but noisier) selection.
            - Change max_depth to None to allow fully-grown trees (may overfit
              and give unstable importances on small datasets).
            - Change class_weight to None if your dataset is balanced.
        """
        # Build the scout RF with balanced class weights so that the minority
        # class (seizure, which is rarer than normal in EEG data) isn't
        # drowned out when computing feature importances.
        # n_jobs=-1 uses ALL available CPU cores to speed up fitting.
        rf_boot = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",   # compensates for seizure being rare
            random_state=random_state, # makes results reproducible
            n_jobs=-1,                 # use all CPU cores; remove if causing issues
        )

        # Actually train the scout RF — this is where most of the compute time goes.
        rf_boot.fit(X, y)

        # feature_importances_ is a 1-D array of floats (one per column in X).
        # Each value is the mean decrease in Gini impurity contributed by that
        # feature across all trees. Higher = more useful for classification.
        # Values sum to 1.0 across all features.
        self.importances_ = rf_boot.feature_importances_

        # --- Select the top-k indices -----------------------------------------
        # Step-by-step breakdown:
        #   np.argsort(self.importances_)
        #       → indices that would sort importances in ASCENDING order (worst first)
        #   [::-1]
        #       → reverse to get DESCENDING order (best first)
        #   [:self.k]
        #       → keep only the first k indices (the k most important features)
        #   np.sort(...)
        #       → re-sort those k indices in ASCENDING order so that when we
        #          do X[:, top_indices_] the columns come out in the SAME LEFT-TO-RIGHT
        #          order as they were in the original feature matrix.
        #          This is important: if you swap columns the model sees a different
        #          feature in each position and will predict garbage.
        self.top_indices_ = np.sort(
            np.argsort(self.importances_)[::-1][:self.k]
        )

        # Return self so callers can chain: selector.fit(X, y).transform(X)
        return self

    # --- 3c. Transform — slice the feature matrix to the top-k columns -------
    def transform(self, X):
        """
        Drop all columns except the top-k selected during fit().

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. Must have the SAME number of columns as the X
            that was passed to fit(); column meaning must also be the same
            (i.e. feature 0 in training must be feature 0 at inference too).

        Returns
        -------
        numpy.ndarray, shape (n_samples, k)
            Feature matrix with only the top-k columns retained, in their
            original left-to-right order.

        HOW TO CHANGE:
            - If your inference pipeline produces features in a different order
              than training, you need to reorder them BEFORE calling transform(),
              not inside this method.
            - Call transform() once per incoming EEG window on the Pi, right
              before passing the window to the main classifier.
        """
        # np.asarray() handles lists, pandas DataFrames, or numpy arrays equally.
        # The fancy indexing [:, self.top_indices_] selects exactly the k columns
        # we decided on during fit(). No copies beyond what numpy needs.
        return np.asarray(X)[:, self.top_indices_]

    # --- 3d. fit_transform — convenience: fit and transform in one call ------
    def fit_transform(self, X, y, **kw):
        """
        Fit the selector on X, then immediately transform X with it.
        Equivalent to calling .fit(X, y).transform(X) but in one line.

        Parameters
        ----------
        X : array-like — full training feature matrix
        y : array-like — class labels
        **kw : extra keyword arguments forwarded to .fit()
               (e.g. n_estimators=100, max_depth=5)

        Returns
        -------
        numpy.ndarray, shape (n_samples, k)
            The training data reduced to the top-k features.

        Typical usage in a training script:
            X_reduced = selector.fit_transform(X_train, y_train, n_estimators=200)
            # Then train the real model on X_reduced
            model.fit(X_reduced, y_train)

        HOW TO CHANGE:
            - Pass keyword args to override the RF defaults:
              selector.fit_transform(X, y, n_estimators=100, max_depth=5)
        """
        # Chain fit then transform.  **kw passes through any extra RF arguments.
        return self.fit(X, y, **kw).transform(X)
