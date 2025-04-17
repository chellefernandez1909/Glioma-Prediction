# library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# load in the dataset
data = pd.read_csv("giloma_grading_clinical_preprocessed_data.csv", index_col= 0)
print(data.head()) # response col == 0

# Baseline model 1: Using logistic regression 
'''
- Using dimensionality reduced (FAMD) dataset
'''
X = data.iloc[:, 1:]
Y = data.iloc[:, 1]
# print(Y)




# logistic regression
# solver = "lbfgs", cause it works for multiclass
# logreg = LogisticRegression(max_iter=2000, solver='lbfgs')
# logreg.fit(x_train, y_train)
# pred_logreg = logreg.predict(x_test)

# # confusion matrix logreg
# logreg_cm = confusion_matrix(y_true= y_test, y_pred= logreg.predict(x_test))
# print("Logistic Regression Confusion Matrix:")
# print(logreg_cm)
# print("Logistic Regression Classification Report:")
# print(classification_report(y_true= y_test, y_pred= logreg.predict(x_test)))

# # Baseline model 2: SVM
# '''
# - Using dimensionality reduced (FAMD) dataset
# - tune for C and gamma
# '''
# # RBF SVM
# svm_rbf = SVC()

# # create param grid
# '''
# C = [50, 10, 1.0, 0.1, 0.01]
# gamma
# '''


# # define grid
# # ref: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
# grid = dict(kernel = ['rbf'], C = C)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=svm_rbf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(x_train_dsampled, y_train_dsampled)

# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) #Best: 0.959067 using {'C': 50, 'kernel': 'rbf'}

# # create final RBF SVM with 'C': 50
# # ref: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# svm_rbf_final = SVC(C = 50, kernel= 'rbf')
# svm_rbf_final.fit(x_train_dsampled, y_train_dsampled)

# # rbf confusion matrix
# svm_rbf_cm = confusion_matrix(y_true= y_test, y_pred= svm_rbf_final.predict(x_test))
# print("RBF SVM (c = 50) Confusion Matrix:")
# print(svm_rbf_cm)
# print("RBF SVM (c = 50) Classification Report:")
# print(classification_report(y_true= y_test, y_pred= svm_rbf_final.predict(x_test)))
