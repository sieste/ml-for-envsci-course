import gzip
import matplotlib.pyplot as plt
import numpy as np


# load tree cover data as in decision-tree code
data_file = "data/covertype/covtype.data.gz"
with gzip.open(data_file, "rt") as f:
    covtype_data = np.genfromtxt(f, delimiter=",")
cover = covtype_data[:, -1]
y = np.zeros_like(cover)
y[ np.isin(cover, [1,6]) ] = 1 # fir


# use columns 0 (elevation) and 1 (aspect) as features; all features are as
# follows: 0: Elevation [m], 1: Aspect [azimuth], 2: Slope [deg], 3: Horz.
# Dist. to Water [m], 4: Vert. Dist. to Water [m], 5: Horz. Dist. to Road [m]
i_feat = [0, 1]
X = covtype_data[:, i_feat]


# fit random forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# cross validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42
)

# Train
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

yclim_pred = np.zeros_like(y_pred)
print("Climatological accuracy:", accuracy_score(y_test, yclim_pred))

# Classification report
# print(classification_report(y_test, y_pred))


# prediction surface over x1/x2 plane vs best decision tree
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 400)
x2 = np.linspace(X[:,1].min(), X[:,1].max(), 400)
xx1, xx2 = np.meshgrid(x1, x2)
grid = np.column_stack([xx1.ravel(), xx2.ravel()])
Z = rf.predict(grid).reshape(xx1.shape)
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.contourf(xx1, xx2, Z, alpha=0.3)
ax.set_title("Decision tree")
ax.set_xlabel("Altitude")
ax.set_ylabel("Aspect")
plt.tight_layout()
plt.show()


# classification probability surface over x1/x2 plane
Z = rf.predict_proba(grid)[:,1].reshape(xx1.shape)
cf = plt.contourf(xx1, xx2, Z, levels=np.linspace(0, 1, 21), vmin=0, vmax=1,)
plt.colorbar(cf, label="probability")
plt.xlabel("Altitude")
plt.ylabel("Aspect")
plt.show()


# fit gradient boosted decision tree with XGBoost
import xgboost as xgb

# target (last column) tree cover encoded as integer 1-7. we coarse-grain the
# tree type into fir (1) and non-fir species (0)
cover = covtype_data[:, -1]
y = np.zeros_like(cover)
y[ np.isin(cover, [1,6]) ] = 1

# use all features; 0: Elevation [m], 1:
# Aspect [azimuth], 2: Slope [deg], 3: Horz. Dist. to Water [m], 4: Vert. Dist.
# to Water [m], 5: Horz. Dist. to Road [m]
i_feat = [0, 1, 2, 3, 4, 5]
X = covtype_data[:, i_feat]


# cross validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)


# simple XGB model (very weak learners, few boosting rounds)
clf_xgb = xgb.XGBClassifier(
    n_estimators = 10,
    max_depth = 1,
    random_state = 42
)
clf_xgb.fit(X_train, y_train)

# predict
y_pred_xgb = clf_xgb.predict(X_test)

# evaluate
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(acc_xgb) # accuracy about .72 



# compare with standard random forest with same number of weak learners
clf_rf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=1)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(acc_rf) # accuracy about 0.6, much lower than boosting


# now train a model on all numeric features (first 6 columns)
X = covtype_data[:, 0:6]

# train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# full random forest
rf_full = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_full.fit(X_train, y_train)
y_pred_rf = rf_full.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Random forest accuracy:", acc_rf)


# boosted trees (XGBoost)
xgb_full = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_full.fit(X_train, y_train)
y_pred_xgb = xgb_full.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print("Boosted tree accuracy:", acc_xgb)


# Exercise: Play with the hyperparameters, see if you can find a model with
# even higher accuracy.


