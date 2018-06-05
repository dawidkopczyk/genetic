import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

SEED = 2018
random.seed(SEED)
np.random.seed(SEED)

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
# Data 
#==============================================================================
dataset = load_boston()
X, y = dataset.data, dataset.target
features = dataset.feature_names

#==============================================================================
# CV MSE before feature selection
#==============================================================================
est = LinearRegression()
score = -1.0 * cross_val_score(est, X, y, cv=5, scoring="neg_mean_squared_error")
print("CV MSE before feature selection: {:.2f}".format(np.mean(score)))

#==============================================================================
# CV MSE after feature selection: RFE
#==============================================================================
rfe = RFECV(est, cv=5, scoring="neg_mean_squared_error")
rfe.fit(X, y)
score = -1.0 * cross_val_score(est, X[:,rfe.support_], y, cv=5, scoring="neg_mean_squared_error")
print("CV MSE after RFE feature selection: {:.2f}".format(np.mean(score)))

#==============================================================================
# CV MSE after feature selection: Feature Importance
#==============================================================================
rf = RandomForestRegressor(n_estimators=500, random_state=SEED)
rf.fit(X, y)
support = rf.feature_importances_ > 0.01
score = -1.0 * cross_val_score(est, X[:,support], y, cv=5, scoring="neg_mean_squared_error")
print("CV MSE after Feature Importance feature selection: {:.2f}".format(np.mean(score)))

#==============================================================================
# CV MSE after feature selection: Boruta
#==============================================================================
rf = RandomForestRegressor(n_estimators=500, random_state=SEED)
boruta = BorutaPy(rf, n_estimators='auto')
boruta.fit(X, y)
score = -1.0 * cross_val_score(est, X[:,boruta.support_], y, cv=5, scoring="neg_mean_squared_error")
print("CV MSE after Boruta feature selection: {:.2f}".format(np.mean(score)))
