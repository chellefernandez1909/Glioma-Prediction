import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
    C=45,             
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

# --- define stacking ensemble ---
# List of base estimators
base_estimators = [
    ('catboost', cb_classifier),
    ('svm', svm_classifier),
    ('random_forest', rf_classifier)
]

# Meta-learner
meta_learner = LogisticRegression(
    solver='lbfgs',
    max_iter=2000,
    random_state=RANDOM_STATE
)

# Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE), 
    stack_method='auto', 
    n_jobs=-1 
)

# Cross-validation 
cv_outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Scoring metrics
num_classes = len(np.unique(y))
average_method = 'binary' if num_classes == 2 else 'macro'

scoring_dict = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average=average_method, zero_division=0),
    'recall': make_scorer(recall_score, average=average_method, zero_division=0),
    'f1': make_scorer(f1_score, average=average_method, zero_division=0),
    'auc': make_scorer(roc_auc_score)
}


# Perform cross-validation
cv_results = cross_validate(
    stacking_classifier,
    X,
    y,
    cv=cv_outer,
    scoring=scoring_dict
)
print("\n--- Cross-Validation Results ---")
for metric in cv_results:
    if metric.startswith('test_'):
        mean_val = np.nanmean(cv_results[metric])
        std_val = np.nanstd(cv_results[metric]) 
        metric_name = metric.split('test_')[1]
        print(f"  {metric_name.capitalize()}: {mean_val:.4f} (+/- {std_val:.4f})")

print("\nStacking ensemble evaluation complete.")