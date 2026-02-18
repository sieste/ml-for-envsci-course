import gzip
import numpy as np
import matplotlib.pyplot as plt


# simple artificial example: decision tree for binary classification

# Generate data
n = 300
X = np.random.uniform(-1, 1, (n, 2))
x1 = X[:,0]
x2 = X[:,1]
y = ((x1 > 0) & (np.abs(x2) < 0.5)).astype(int)

# Plot true rule and sample points
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X[:,0], X[:,1], c=y, s=20)
# decision boundaries: x1 = 0 and x2 = +/-0.5
ax.axvline(x=0, ymin=0.25, ymax=0.75, linestyle='--', color='black')
ax.axhline(y=0.5, xmin=0.5, xmax=1, linestyle='--', color='black')
ax.axhline(y=-0.5, xmin=0.5, xmax=1, linestyle='--', color='black')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


# train the decision tree in scikit-learn
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X,y)


# visualise decision surface
x0, x1 = np.mgrid[-1:1:100j, -1:1:100j]
grid = np.column_stack([x0.ravel(), x1.ravel()])
Z = clf.predict(grid).reshape(x0.shape)
plt.contourf(x0, x1, Z, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, s=20)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# exercise: reduce/increase max_depth, how does the decision surface change?



# visualising a shallow decision tree
from sklearn import tree
plt.figure(figsize=(10, 6))
tree.plot_tree(clf,
    feature_names=["x1", "x2"],
    class_names=["0", "1"],
    filled=True, rounded=True,
    fontsize=8
)
plt.show()



# ICS tree data set
# url = "https://archive.ics.uci.edu/static/public/31/covertype.zip"

# unzip and load into numpy array
data_file = "data/covertype/covtype.data.gz"
with gzip.open(data_file, "rt") as f:
    covtype_data = np.genfromtxt(f, delimiter=",")

# column names are documented in covtype.info

# target (last column) tree cover encoded as integer 1-7. we coarse-grain the
# tree type into fir (1) and non-fir species (0)
cover = covtype_data[:, -1]
y = np.zeros_like(cover)
y[ np.isin(cover, [1,6]) ] = 1 # fir


np.mean(y)
# in total, about 40% of trees in the data set are firs. 
# so a default "best-guess" prediction could be "no fir" 


# use columns 0 (elevation) and 1 (aspect) as features; all features are as
# follows: 0: Elevation [m], 1: Aspect [azimuth], 2: Slope [deg], 3: Horz.
# Dist. to Water [m], 4: Vert. Dist. to Water [m], 5: Horz. Dist. to Road [m]
i_feat = [0, 1]
X = covtype_data[:, i_feat]
i_sub = np.random.randint(0, len(X), 5000)
Xs, ys = X[i_sub, :], y[i_sub]
for i in [0,1]:
  cover = 'non-fir' if i == 0 else 'fir'
  mask = (ys == i)
  plt.plot(Xs[mask,0], Xs[mask, 1], '.', label=cover, alpha=.5)
plt.legend()
plt.title('Subset of 5000 trees from CoverType data')
plt.xlabel('Elevation')
plt.ylabel('Aspect')
plt.show()

# difficult to see a difference from scatter plot 



# fit a small decision tree
from sklearn.tree import DecisionTreeClassifier

# X: shape (n_samples, 2) with columns [altitude, aspect]
# y: shape (n_samples,) with 0/1 labels for "fir"

clf = DecisionTreeClassifier(
    max_depth=10   # small tree for easy visualization
)
clf.fit(X, y)


# visualise classification over x1/x2 grid
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 400)
x2 = np.linspace(X[:,1].min(), X[:,1].max(), 400)
xx1, xx2 = np.meshgrid(x1, x2)
grid = np.column_stack([xx1.ravel(), xx2.ravel()])
yhat_grid = clf.predict(grid)
Z = yhat_grid.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3)
# plt.scatter(X[:10000,0], X[:10000,1], c=y[:10000], s=1, alpha=.1)
plt.xlabel("Altitude")
plt.ylabel("Aspect")
plt.show()


# visualise probability P(y=1) over x1/x2 grid
yhat_prob_grid = clf.predict_proba(grid) # (n, 2) array with probs for y=0 and y=1
Z = yhat_prob_grid[:,1].reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3)
# plt.scatter(X[:10000,0], X[:10000,1], c=y[:10000], s=1, alpha=.1)
plt.xlabel("Altitude")
plt.ylabel("Aspect")
plt.colorbar()
plt.show()


# visualising a shallow decision tree
from sklearn import tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
plt.figure(figsize=(10, 6))
tree.plot_tree(clf,
    feature_names=["altitude", "aspect"],
    class_names=["not_fir", "fir"],
    filled=True, rounded=True,
    fontsize=8
)
plt.show()




# decision surfaces for different tree depths
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 400)
x2 = np.linspace(X[:,1].min(), X[:,1].max(), 400)
xx1, xx2 = np.meshgrid(x1, x2)
grid = np.column_stack([xx1.ravel(), xx2.ravel()])
maxdepths = [4, 8, 16]
Z = []
for md in maxdepths:
  clf = DecisionTreeClassifier(max_depth=md)
  clf.fit(X, y)
  Z.append(clf.predict(grid).reshape(xx1.shape))
fig,axs = plt.subplots(len(Z),1,figsize=(4,7))
for i,md in enumerate(maxdepths):
  axs[i].contourf(xx1, xx2, Z[i], alpha=0.3)
  axs[i].set_title(f"max_depth = {md}")
plt.tight_layout()
plt.show()




# list all model hyperparameters

clf.get_params()

# {'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': 16,
#  'max_features': None,
#  'max_leaf_nodes': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'monotonic_cst': None,
#  'random_state': None,
#  'splitter': 'best'}

# hyper parameter optimisations
from sklearn.model_selection import GridSearchCV
clf_cv = GridSearchCV(
    estimator=clf,
    param_grid={"max_depth": [4,8,16,32], "criterion": ["gini", "entropy"]},
    cv=5
)
clf_cv.fit(X, y)

# predict
Z = clf_cv.predict(grid).reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3)
plt.xlabel("Altitude")
plt.ylabel("Aspect")
plt.show()

# predict probability
Z = clf_cv.predict_proba(grid)[:,1].reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3)
plt.xlabel("Altitude")
plt.ylabel("Aspect")
plt.show()

