# library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# load in the dataset
data = pd.read_csv("giloma_grading_clinical_preprocessed_data.csv", index_col= 0)
# print(data.head()) # response col == 0

#=================================================
# Baseline model 1: Using logistic regression 
#=================================================
# define X (preds) and Y (response)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

# logistic regression
logreg = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight = 'balanced')

# 10 fold k-f cv
kfcv = StratifiedKFold(n_splits= 10, shuffle = True, random_state= 6740)

# metrics
lg_precision_score = []
lg_recall_score = []
lg_f1_score = []
lg_auc_score = []


for fold, (train_idx, test_idx) in enumerate(kfcv.split(X, Y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_proba = logreg.predict_proba(X_test)[:, 1]  # for AUC

    # Calculate and store metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    lg_precision_score.append(precision)
    lg_recall_score.append(recall)
    lg_f1_score.append(f1)
    lg_auc_score.append(auc)

    print(f"\nFold {fold}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

# Final results
print("\n=== Average Cross-Validated Metrics Logistic Regression (10 Folds) ===")
print(f"Avg Precision: {np.mean(lg_precision_score):.3f}")
print(f"Avg Recall:    {np.mean(lg_recall_score):.3f}")
print(f"Avg F1 Score:  {np.mean(lg_f1_score):.3f}")
print(f"Avg AUC:       {np.mean(lg_auc_score):.3f}")

#=================================================
# Baseline model 2: SVM
#=================================================
'''
# ref: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
- Using dimensionality reduced (FAMD) dataset
- tune for C and gamma
'''
# ===============
# Tune best C and gamma with 10-fold CV using AUC
gamma_range = list(np.arange(0.01, 1.01, 0.01)) + ['scale']
param_grid = {
    'C': [50, 45, 35, 25, 10, 1.0, 0.1, 0.01],
    'gamma': gamma_range,
    'kernel': ['rbf'],
    'class_weight': ['balanced']
}

outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state= 6740)

grid = GridSearchCV(
    estimator=SVC(probability=True),
    param_grid=param_grid,
    cv=outer_cv,
    scoring='roc_auc',
    n_jobs=-1
)

# print("Tuning best hyperparameters (C, gamma)...")
grid.fit(X, Y)
best_params = grid.best_params_

# best params
best_c = best_params['C'] 
best_gamma = best_params['gamma']

print(f"\n Best Hyperparameters Found:\nC = {best_c}, gamma = {best_gamma}")

# Using best hyperparameters, perform 10-fold CV 
svm_precision_score = []
svm_recall_score = []
svm_f1_score = []
svm_auc_score = []

model = SVC(
    kernel='rbf',
    C= best_c,
    gamma= best_gamma,
    class_weight='balanced',
    probability=True
)

kfcv = StratifiedKFold(n_splits=10, shuffle=True, random_state=6740)

for fold, (train_idx, test_idx) in enumerate(kfcv.split(X, Y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    svm_precision_score.append(precision)
    svm_recall_score.append(recall)
    svm_f1_score.append(f1)
    svm_auc_score.append(auc)

    print(f"\nFold {fold} Metrics:")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

# Step 3: Report overall metrics
print("\n=== Final 10-Fold Cross-Validated Performance with Tuned RBF-SVM ===")
print(f"Avg Precision: {np.mean(svm_precision_score):.3f}")
print(f"Avg Recall:    {np.mean(svm_recall_score):.3f}")
print(f"Avg F1 Score:  {np.mean(svm_f1_score):.3f}")
print(f"Avg AUC:       {np.mean(svm_auc_score):.3f}")
