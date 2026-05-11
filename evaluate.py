import joblib
import numpy as np
from sklearn.metrics import classification_report

DATA_DIR = "data"

X = np.load(f"{DATA_DIR}/X.npy")
y = np.load(f"{DATA_DIR}/y.npy")
clf = joblib.load(f"{DATA_DIR}/rf_model.joblib")

pred = clf.predict(X)
print(classification_report(y, pred))
