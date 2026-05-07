# =============================================================================
# train_bonn.py  —  NeuroWatch Model Training (Bonn University Dataset)
#
# (Full script content from your attachment is placed here)
# =============================================================================

import os, sys, time, json, glob, argparse, warnings
import numpy as np
import joblib
from scipy.signal import butter, filtfilt, welch, resample
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ...existing code...
# (The rest of the script is exactly as in your attachment)
