import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from prince import FAMD

# --- Configuration ---
FILE_PATH = 'giloma_grading_clinical_data_rf_catboost.csv'
TARGET_COLUMN = 'Grade'
N_SPLITS = 10 # Number of folds for cross-validation
RANDOM_STATE = 1942

# --- Load Data ---
df = pd.read_csv(FILE_PATH)

columns_to_drop = [TARGET_COLUMN, 'Primary_Diagnosis']
X = df.drop(columns=columns_to_drop)
y = df[TARGET_COLUMN]

# --- Encoding Categorical Columns ---
categorical_cols_to_encode = [
    'Gender', 'Race', 'IDH1', 'TP53', 'ATRX', 'PTEN',
    'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1',
    'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']

# Filter list to include only columns present in X
categorical_cols_present = [col for col in categorical_cols_to_encode if col in X.columns]
for col in categorical_cols_present:
    X[col], _ = pd.factorize(X[col])

# FAMD 
print("FAMD start")
print()

giloma_famd = FAMD(n_components=20, random_state=6740)
giloma_famd = giloma_famd.fit(X)
X_famd = giloma_famd.transform(X)

# Assess the variance explained by each component (eigevalues)
eigenvals = giloma_famd.eigenvalues_

top_3_col_components = X_famd.iloc[:,:3]

# print("Top 3 components")
# print(top_3_col_components)

print()
print("FAMD done")

top_3_x_merged = pd.concat([X, top_3_col_components], axis=1)

# print("Merged df")
# print(top_3_x)

svm_features = top_3_col_components.columns.to_list()
other_features = X.columns.to_list()

print("Other features")
print(other_features)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names]
    

catboost_cat_features = categorical_cols_present

# --- Base Model Definitions ---
#CatBoost Classifier
cb_classifier = CatBoostClassifier(
    iterations=50, 
    depth=8,        
    learning_rate=0.05,
    l2_leaf_reg=3,    
    random_state=RANDOM_STATE,
    cat_features=catboost_cat_features,
    verbose=0, 
    thread_count=6 
)

# SVM Classifier
svm_classifier = SVC(
    C=1,             
    gamma='scale',        
    kernel='rbf',     
    probability=True, 
    class_weight='balanced', 
    random_state=RANDOM_STATE
)

#RF Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,        
    max_depth=15,            
    min_samples_split=10,     
    min_samples_leaf=1,     
    max_features='sqrt',     
    bootstrap=True,          
    random_state=RANDOM_STATE
)

'''
#Logistic Regression Classifier
logreg_classifier = LogisticRegression(
    max_iter=2000,         
    solver='lbfgs',        
    class_weight='balanced', 
    random_state=RANDOM_STATE
)
'''

# --- Define pipelines for each base model with appropiate feature subsets ---

# CatBoost pipelien: uses original features (excluding FAMD components)
cb_pipeline = Pipeline([
    ('selector', FeatureSelector(feature_names=other_features)),
    ('cb', cb_classifier)
])

# SVM pipeline: uses top 3 FAMD components for dimensionality-reduced input
svm_pipeline = Pipeline([
    ('selector', FeatureSelector(feature_names=svm_features)),
    ('svm', svm_classifier)
])

# Random Forest pipeline: also uses original features set
rf_pipeline = Pipeline([
    ('selector', FeatureSelector(feature_names=other_features)),
    ('rf', rf_classifier)
])


# --- define stacking ensemble ---
# List of base estimators
base_estimators = [
    ('catboost', cb_pipeline),
    ('svm', svm_pipeline),
    ('random_forest', rf_pipeline)
]

# Meta-learner
meta_learner_base = LogisticRegression(
    solver='lbfgs',
    max_iter=2000,
    #class_weight='balanced',
    random_state=RANDOM_STATE
)

# Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner_base,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE), 
    stack_method='auto', 
    n_jobs=-1 
)

param_grid_stacking = {
    'final_estimator__C': [0.01, 0.1, 1, 10, 100],
    'final_estimator__class_weight': [None, 'balanced']
}

# Cross-validation 
cv_outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Scoring metrics
num_classes = len(np.unique(y))
average_method = 'binary' if num_classes == 2 else 'macro'
main_scorer_name = 'f1' 
main_scorer = make_scorer(f1_score, average=average_method, zero_division=0)
scoring_dict = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average=average_method, zero_division=0),
    'recall': make_scorer(recall_score, average=average_method, zero_division=0),
    'f1': make_scorer(f1_score, average=average_method, zero_division=0),
    'auc': 'roc_auc' 
}


grid_search_stacking = GridSearchCV(
    estimator=stacking_classifier,
    param_grid=param_grid_stacking,
    scoring=main_scorer_name, 
    cv=cv_outer,              
    n_jobs=-1,                
    verbose=1                 
)

# --- Fit GridSearchCV ---
print(f"\n--- Tuning Stacking Classifier (Meta-Learner) using {N_SPLITS}-Fold CV ---")
grid_search_stacking.fit(top_3_x_merged, y)


# --- Tuning Results ---
print("\n--- Tuning Results ---")
print("Best Parameters Found for Stacking Classifier (Meta-Learner):")
for param, value in grid_search_stacking.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest Cross-Validation Score ({main_scorer_name.upper()}):")
print(f"  {grid_search_stacking.best_score_:.4f}")

# --- Evaluate the BEST Stacking Classifier found by GridSearchCV ---
print("\n--- Evaluating Best Stacking Classifier with Cross-Validation ---")
best_stacker = grid_search_stacking.best_estimator_



cv_results_final = cross_validate(
    best_stacker,
    top_3_x_merged,
    y,
    cv=cv_outer,
    scoring=scoring_dict,
    n_jobs=-1,
    error_score='raise'
)


print("\nFinal Cross-Validation Metrics (Mean +/- Std Dev):")
for metric_key in scoring_dict.keys():
    test_score_key = f'test_{metric_key}'
    if test_score_key in cv_results_final:
        mean_val = np.nanmean(cv_results_final[test_score_key])
        std_val = np.nanstd(cv_results_final[test_score_key])
        print(f"  {metric_key.capitalize()}: {mean_val:.4f} (+/- {std_val:.4f})")
    else:
            print(f"  {metric_key.capitalize()}: Metric not found in results.")


print("\nStacking ensemble tuning and evaluation complete.")