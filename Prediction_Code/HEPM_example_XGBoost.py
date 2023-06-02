# An example of REPM when compiling the CEEG dataset
# Authors: Zonghan Li, Chunyan Wang, Yi Liu
# Package importing
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Data importing
df = pd.read_csv('…')
df.head()
df.info()

df_pred = pd.read_csv('…')
df_pred.head()
df_pred.info()

prob = []

# Model establishment
y = df.iloc[:, 7]
X = df.iloc[:, 0:7]
X_pred = df_pred.iloc[:, 0:7]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
xgbc = xgb.XGBClassifier(learning_rate =0.01, n_estimators=350, max_depth=3,
                       min_child_weight=5, gamma=0.4, subsample=0.75,
                       colsample_bytree=0.75,
                       objective= 'binary:logistic', nthread=4,
scale_pos_weight=1)
xgbc.fit(X_train, y_train)

# Key features identification
importances = xgbc.feature_importances_
feat_labels = X.columns[0:]
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

# Threshold Calculation
y_pred = xgbc.predict(X_test)
y_prob = xgbc.predict_proba(X_test)
y_prob = xgbc.predict_proba(X_test)[:, 1]
FPR, TPR, thresholds = metrics.roc_curve(y_test, y_prob)
maxindex = (TPR - FPR).tolist().index(max(TPR - FPR))
threshold = thresholds[maxindex]
y_pred1 = (xgbc.predict_proba(X_test)[:, 1] >= threshold).astype(int)
y_prediction_prob = xgbc.predict_proba(X_pred)[:, 1]

# Visualizing the results
y_test_fig = np.array(y_test)
plt.plot(y_test_fig, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

# Performance evaluation. Note than the model was repeated for 100 times and the results of the best 20 times were used.
pr1 = metrics.precision_score(y_test, y_pred1)  # Precision
re1 = metrics.recall_score(y_test, y_pred1) # Recall
f11 = metrics.f1_score(y_test, y_pred1) # F1 score

# An example of GridSearch
from sklearn.model_selection import GridSearchCV
param_test_gamma={
    'gamma':[i/10.0 for i in range(0,5)],
}
gsearch_gamma = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.02, n_estimators=350, max_depth=3,
 min_child_weight=5, gamma=0.4, subsample=0.8, colsample_bytree=0.75,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=81),
 param_grid = param_test_gamma, scoring='f1',n_jobs=4, cv=5)
gsearch_gamma.fit(X_train, y_train)
gsearch_gamma.best_params_, gsearch_gamma.best_score_

