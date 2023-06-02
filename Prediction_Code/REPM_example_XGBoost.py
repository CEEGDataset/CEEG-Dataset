# An example of REPM when compiling the CEEG dataset
# Authors: Zonghan Li, Chunyan Wang, Yi Liu
# Package importing
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Data importing
df = pd.read_csv('…')
df.head()
df.info()

df_pred = pd.read_csv('…')
df_pred.head()
df_pred.info()

# Model establishment
y = df.iloc[:, 8]
X = df.iloc[:, 0:8]
X_pred = df_pred.iloc[:, 0:8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
xgbr = xgb.XGBRegressor(learning_rate =0.1, n_estimators=350, max_depth=4,
                        min_child_weight=4, gamma=0.0, subsample=0.85,
                        colsample_bytree=0.75,
                        objective= 'reg:gamma', nthread=4)
xgbr.fit(X_train, y_train)
y_pred = xgbr.predict(X_test)

# Key features identification.
importances = linear_svr.feature_importances_
feat_labels = X.columns[0:]
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

# Performance evaluation
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
R2 = metrics.r2_score(y_test, y_pred)
MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)

# Visualizing the results
y_test_fig = np.array(y_test)
plt.plot(y_test_fig, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

# Prediction. Note than the model was repeated for 100 times and the results of the best 20 times were used.
y_predset = xgbr.predict(X_pred)

# An example of GridSearch
param_test_md_mcw={
    'max_depth':list(range(3,10)),
    'min_child_weight':list(range(1,6))
}
gsearch_md_mcw = GridSearchCV(estimator = xgb.XGBRegressor(learning_rate =0.1, n_estimators=350, max_depth=4, min_child_weight=4, gamma=0.0, subsample=0.85, colsample_bytree=0.75, objective= 'reg:gamma', nthread=4),
 param_grid = param_test1, scoring='neg_root_mean_squared_error',n_jobs=20,cv=5)
gsearch_md_mcw.fit(X_train, y_train)
gsearch_md_mcw.best_params_, gsearch_md_mcw.best_score_
