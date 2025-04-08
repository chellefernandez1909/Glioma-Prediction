# team109
This is a group project for a course project. 

The dataset used for the project is the Glioma dataset from the UCI machine learning repository.
link: https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset

Research question: 
Compare various machine learning models for predicting whether a patient has LGG or GBM.

Analysis planned:

**EDA:**
-  check data uniformly labeled
-  checking missing data (imputation if needed)
-  Cramer V/ point biserial for multicollinearity
-  check class imbalance (bar plot)
-  one-hot encoding categorical variables
-  FAMD (visual analysis, relationship between categorical variables)
-  Variable selection (FAMD? Lasso/ Elastic Net)

**Handling class imbalance**
- SMOTE-NC/ Downsampling/ Hybrid/ EasyEnsemble/ Balanced Random Forest
- compare sample generated using bootstrapping to figure out which sample is best

(notes)
- Logistic regresion : sensitive to sampling method
- SVM: smote maybe better, best compare
- RF: can hanfle imbalance, downsampling may be better, best compare
- catBoost: class_weights/ compare models

Models
(with var selected df)
- Logistic regression (baseline) + top predictors 
- RBF SVM (tune C and gamma)

(full df)
- Random Forest (tune based on F1 score and AUC, tune num_features, max_depth, number of trees, min sample at each node) + SHAP
- catBoost (depth, learning rate, L2 regularization) + SHAP
- stacking ensemble

Cross validation 
- 10-fold cross validation
- mean and variance of testing error
- precision, recall, F1-score, AUC
