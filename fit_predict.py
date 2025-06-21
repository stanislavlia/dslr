#!/usr/bin/env python3


import sys
import pandas as pd
import numpy as np
from logistic_regression import OVALogisticRegression #OUR IMPLEMENTATION 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


if len(sys.argv) != 2:
    print("Usage: describe.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
try:
    with open('train_vif_elim.csv', 'r') as file:
        df = pd.read_csv(file).drop(columns='Index')
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)



X_train_reduced_scaled = df.drop(columns='Hogwarts House')
y_train_encoded = df['Hogwarts House']


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scores = []

y_train_encoded = y_train_encoded.to_numpy().reshape(-1, 1)

for train_idx, test_idx in cv.split(X_train_reduced_scaled, y_train_encoded):
    X_tr, X_te = X_train_reduced_scaled.iloc[train_idx].to_numpy(), X_train_reduced_scaled.iloc[test_idx].to_numpy()
    y_tr, y_te = y_train_encoded[train_idx],      y_train_encoded[test_idx]

    # fresh classifier each fold
    clf = OVALogisticRegression(lr=0.01, iterations=10000, l2_penalty=0.00)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    scores.append(accuracy_score(y_te, y_pred))

scores = np.array(scores)
print(f"CV accuracies: {scores}")
print(f"Mean CV accuracy: {scores.mean():.5f} ± {scores.std():.3f}")
print(f"Interval = [{scores.mean() - scores.std():.5f}, {scores.mean() + scores.std():.5f}]")