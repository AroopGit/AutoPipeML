import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

# Best pipeline found by AutoML
best_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('estimator', DecisionTreeClassifier(max_depth=13, random_state=42)),
])

# Usage:
# pipeline.fit(X_train, y_train)
# predictions = pipeline.predict(X_test)