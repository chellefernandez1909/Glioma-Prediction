import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate

# --- Configuration ---
FILE_PATH = 'giloma_grading_clinical_data_rf_catboost.csv'
TARGET_COLUMN = 'Grade'
N_SPLITS = 10 #folds for cv
RANDOM_STATE = 1942

# --- Load Data ---
df = pd.read_csv(FILE_PATH)

# --- Feature/Target Separation ---
X = df.drop(columns=[TARGET_COLUMN, 'Primary_Diagnosis'])
y = df[TARGET_COLUMN]


# --- Encoding columns ---
categorical_cols_to_encode = [
    'Gender', 'Primary_Diagnosis', 'Race', 'IDH1', 'TP53', 'ATRX', 'PTEN',
    'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1',
    'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
categorical_cols_to_encode = [col for col in categorical_cols_to_encode if col in X.columns]

'''
for col in categorical_cols_to_encode:
    X[col], _ = pd.factorize(X[col])
'''
    
print(X.head())
#print(y.head())

# ---Model Definition---
cb_classifier = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, cat_features=categorical_cols_to_encode, thread_count=6)

# --- Hyperparameter Tuning ---
param_grid = {
    'iterations': [20, 50, 75, 100],
    'depth': [2, 4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5]
}

#cv
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

#scoring metric
num_classes = len(np.unique(y))
average_method = 'binary' if num_classes == 2 else 'macro'
scoring_metric_name = f'f1_{average_method}' if num_classes > 2 else 'f1'
main_scorer = make_scorer(f1_score, average=average_method, zero_division=0)


n_combinations = 1
for key in param_grid:
    n_combinations *= len(param_grid[key])

#grid search setup
grid_search = GridSearchCV(
    estimator=cb_classifier,
    param_grid=param_grid,
    scoring=main_scorer,
    cv=cv,
    verbose=2
)

#optimal hyperparameters search
print(f"\nSearching for optimal hyperparameters using Grid Search with {N_SPLITS}-fold cross-validation:")

grid_search.fit(X, y, cat_features=categorical_cols_to_encode)

#Results
print(f"Best Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

#best score index
best_index = grid_search.best_index_

mean_score = grid_search.cv_results_['mean_test_score'][best_index]
std_score = grid_search.cv_results_['std_test_score'][best_index]

#print best scores
print(f"\nBest Cross-Validation Score ({scoring_metric_name.replace('_', ' ')}):")
print(f"  {grid_search.best_score_:.4f}")
print("\ntuning and evaluation complete.")


# --- After finding best parameters ---
best_cb = grid_search.best_estimator_
print("\n--- Evaluate best estimator with CV ---")

# socrers
scoring_dict = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average=average_method, zero_division=0),
    'recall': make_scorer(recall_score, average=average_method, zero_division=0),
    'f1': make_scorer(f1_score, average=average_method, zero_division=0),
    'auc': make_scorer(roc_auc_score)
}


cv_results_final = cross_validate(best_cb, X, y, cv=cv, scoring=scoring_dict, error_score=np.nan)
print("\nFinal cv metrics (Mean +/- Std Dev):")
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
if hasattr(best_cb, 'feature_importances_'):
    print("\n--- Feature Importances (from best model) ---")
    importances = best_cb.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(feature_importance_df.head(5))

