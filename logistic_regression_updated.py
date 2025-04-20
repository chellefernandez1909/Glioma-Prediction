# library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from prince import FAMD

# load in the dataset
data = pd.read_csv("giloma_grading_clinical_preprocessed_data.csv", index_col= 0)
# print(data.head()) # response col == 0

#=================================================
# Load dataset 
#=================================================

# define X (preds) and Y (response)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

#=================================================
# Apply FAMD 
#=================================================

print("FAMD start")
print()

giloma_famd = FAMD(n_components=20, random_state=6740)
giloma_famd = giloma_famd.fit(X)
X_famd = giloma_famd.transform(X)

# Assess the variance explained by each component (eigevalues)
eigenvals = giloma_famd.eigenvalues_

# Compute the cumulative explained variance
explained_var = eigenvals / eigenvals.sum()
cum_var = explained_var.cumsum()

# Find number of components that explain at least 95% of the variance
num_of_components_95 = (cum_var >= 0.95).sum() + 1
print(f"Number of components that explain at least 95% of the variance: {num_of_components_95}")

# Getting the top 10 variables by contributions and converting the contributions in percentages for easier interpretability
# Columns contributions determine how much each variable contributes to each component
giloma_famd_col_contributions = giloma_famd.column_contributions_

total_var_contributions = giloma_famd_col_contributions.sum(axis=1)

top_10_vars = total_var_contributions.sort_values(ascending=False).head(10) * 100

top_10_vars = top_10_vars.round(2).astype(str) + '%'

print("Top 10 Variables - by % of variance explained")
print(top_10_vars)

print()
print(f"FAMD done!")

# Use transformed features
X = X_famd

#=================================================
# Baseline model 1: Using logistic regression 
#=================================================

# logistic regression
logreg = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight = 'balanced')

# --- Hyperparameter Tuning ---
logreg_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'max_iter': [500, 1000, 2000]
}

# 10 fold k-f cv
kfcv = StratifiedKFold(n_splits=10, shuffle=True, random_state=6740)

# Detemrine scoring metrics
num_classes = len(np.unique(Y))
average_method = 'binary' if num_classes == 2 else 'macro'
scoring_metric_name = f'f1_{average_method}' if num_classes > 2 else 'f1'
main_scorer = make_scorer(f1_score, average=average_method, zero_division=0)

# Show number of parameter combinations
n_combinations = 1
for key in logreg_param_grid:
    n_combinations *= len(logreg_param_grid[key])

#grid search setup
logreg_grid_search = GridSearchCV(
    estimator=logreg,
    param_grid=logreg_param_grid,
    scoring=main_scorer,
    cv=kfcv,
    n_jobs=-1,
    verbose=1
)

#optimal hyperparameters search
print(f"\nSearching for optimal hyperparameters using Grid Search with {kfcv.get_n_splits()}-fold cross-validation:")
logreg_grid_search.fit(X, Y)

#Results
print(f"Best Parameters Found:")
for param, value in logreg_grid_search.best_params_.items():
    print(f"  {param}: {value}")

#best score index
best_index = logreg_grid_search.best_index_
mean_score = logreg_grid_search.cv_results_['mean_test_score'][best_index]
std_score = logreg_grid_search.cv_results_['std_test_score'][best_index]

#print best scores
print(f"\nBest Cross-Validation Score ({scoring_metric_name.replace('_', ' ')}):")
print(f"  {logreg_grid_search.best_score_:.4f}")
print("\ntuning and evaluation complete.")

# --- After finding best parameters ---
best_logreg = logreg_grid_search.best_estimator_
print("\n--- Evaluate best Logistic Regression estimator with CV ---")

# Define Evaluation metrics
scoring_dict = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average=average_method, zero_division=0),
    'recall': make_scorer(recall_score, average=average_method, zero_division=0),
    'f1': make_scorer(f1_score, average=average_method, zero_division=0),
    'auc': make_scorer(roc_auc_score)
}


cv_results_final = cross_validate(best_logreg, X, Y, cv=kfcv, scoring=scoring_dict, n_jobs=-1, error_score=np.nan)
print(cv_results_final.keys())

print("\nFinal cv metrics for Logistic Regression (Mean +/- Std Dev):")
for metric in cv_results_final:
    if metric.startswith('test_'):
        mean_val = np.mean(cv_results_final[metric])
        std_val = np.std(cv_results_final[metric])
        metric_name = metric.split('test_')[1]
        if np.isnan(mean_val):
             print(f"  {metric_name.capitalize()}: Calculation Failed (NaN)")
        else:
             print(f"  {metric_name.capitalize()}: {mean_val:.4f} (+/- {std_val:.4f})")

#Feature importance
if hasattr(best_logreg, 'feature_importances_'):
    print("\n--- Feature Importances (from best model) ---")
    importances = best_logreg.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(feature_importance_df.head(5))


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
# Define SVM model
svm = SVC(probability=True, class_weight='balanced')

# Tune best C and gamma with 10-fold CV using AUC
gamma_range = list(np.arange(0.01, 1.01, 0.01)) + ['scale']
svm_param_grid = {
    'C': [50, 45, 35, 25, 10, 1.0, 0.1, 0.01],
    'gamma': gamma_range,
    'kernel': ['rbf'],
    'class_weight': ['balanced']
}

# Cross Validation
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=6740)

# Scoring metric
num_classes = len(np.unique(Y))
average_method = 'binary' if num_classes == 2 else 'macro'
scoring_metric_name = f'f1_{average_method}' if num_classes > 2 else 'f1'
main_scorer = make_scorer(f1_score, average=average_method, zero_division=0)

# Show number of parameter combinations
n_combinations = 1
for key in svm_param_grid:
    n_combinations *= len(svm_param_grid[key])

svm_grid_search = GridSearchCV(
    estimator=svm,
    param_grid=svm_param_grid,
    cv=outer_cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

#optimal hyperparameters search
print(f"\nSearching for optimal hyperparameters using Grid Search with {outer_cv.get_n_splits()}-fold cross-validation:")
svm_grid_search.fit(X, Y)

#Results
print(f"Best Parameters Found:")
for param, value in svm_grid_search.best_params_.items():
    print(f"  {param}: {value}")

#best score index
best_index = svm_grid_search.best_index_
mean_score = svm_grid_search.cv_results_['mean_test_score'][best_index]
std_score = svm_grid_search.cv_results_['std_test_score'][best_index]

#print best scores
print(f"\nBest Cross-Validation Score ({scoring_metric_name.replace('_', ' ')}):")
print(f"  {svm_grid_search.best_score_:.4f}")
print("\ntuning and evaluation complete.")

# --- After finding best parameters ---
best_svm = svm_grid_search.best_estimator_
print("\n--- Evaluate best SVM estimator with CV ---")

# Define Evaluation metrics
svm_scoring_dict = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average=average_method, zero_division=0),
    'recall': make_scorer(recall_score, average=average_method, zero_division=0),
    'f1': make_scorer(f1_score, average=average_method, zero_division=0),
    'auc': make_scorer(roc_auc_score)
}

cv_results_final = cross_validate(best_svm, X, Y, cv=outer_cv, scoring=svm_scoring_dict, n_jobs=-1, error_score=np.nan)
print(cv_results_final.keys())

print("\nFinal cv metrics for SVM (Mean +/- Std Dev):")
for metric in cv_results_final:
    if metric.startswith('test_'):
        mean_val = np.mean(cv_results_final[metric])
        std_val = np.std(cv_results_final[metric])
        metric_name = metric.split('test_')[1]
        if np.isnan(mean_val):
             print(f"  {metric_name.capitalize()}: Calculation Failed (NaN)")
        else:
             print(f"  {metric_name.capitalize()}: {mean_val:.4f} (+/- {std_val:.4f})")

#Feature importance
if hasattr(best_svm, 'feature_importances_'):
    print("\n--- Feature Importances (from best model) ---")
    importances = best_logreg.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(feature_importance_df.head(5))
