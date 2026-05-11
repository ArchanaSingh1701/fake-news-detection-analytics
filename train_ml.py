import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = np.load("data/X.npy")
y = np.load("data/y.npy")

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X, y)

pred = clf.predict(X)
print(classification_report(y, pred))

joblib.dump(clf, "data/rf_model.joblib")
print("✅ Model trained")

