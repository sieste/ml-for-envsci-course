import os
import requests
from zipfile import ZipFile
from io import BytesIO
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

# Plot true rule + sample points
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X[:,0], X[:,1], c=y, s=20)
# Draw boundaries: x1 = 0 and x2 = +/-0.5
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
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)


# visualise decision surface
x1_grid = np.linspace(-1, 1, 100)
x2_grid = np.linspace(-1, 1, 100)
xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = clf.predict(grid).reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, s=20)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# exercise: reduce/increase max_depth, how does the decision surface change?


# ICS tree data set
url = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
target_dir = "data/covertype" # change as needed

# download and unzip directly from memory, skip if exists
if not os.path.isdir(target_dir):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with ZipFile(BytesIO(resp.content)) as z:
        z.extractall(path=target_dir)
    print(f"Downloaded and extracted to {target_dir}")
else:
    print(f"{target_dir} exists - skipping download/unzip")

# unzip and load into numpy array
path = f"{target_dir}/covtype.data.gz"  # adjust as needed
with gzip.open(path, "rt") as f:
    arr = np.genfromtxt(f, delimiter=",")

# column names are documented in covtype.info

# target (last column) tree cover encoded as integer 1-7. we coarse-grain the
# tree type into pine (1) and non-pine (0)
cover = arr[:, -1]
y = np.zeros_like(cover)
#y[ np.isin(cover, [2,3]) ] = 1 # pine
y[ np.isin(cover, [1,6]) ] = 1 # fir

# use columns 0 (elevation) and 1 (aspect) as features 0: Elevation [m], 1:
# Aspect [azimuth], 2: Slope [deg], 3: Horz. Dist. to Water [m], 4: Vert. Dist.
# to Water [m], 5: Horz. Dist. to Road [m]
i_feat = [0, 1]
X = arr[:, i_feat]
for i,cover in zip([0,1], ['non-fir', 'fir']):
  mask = (y == i)
  Xi = X[mask,:]
  i_sub = np.random.randint(0, len(Xi), 5000)
  plt.plot(Xi[i_sub,0], Xi[i_sub, 1], '.', label=cover)
plt.legend()
plt.show()

# overall, about 40% firs in the data, firs seem to prefer higher altitudes and
# less exposure to direct sunlight



# fit a small decision tree
from sklearn.tree import DecisionTreeClassifier

# X: shape (n_samples, 2) with columns [altitude, aspect]
# y: shape (n_samples,) with 0/1 labels for "fir"

clf = DecisionTreeClassifier(
    max_depth=10   # small tree for easy visualization
)
clf.fit(X, y)


# grid
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 400)
x2 = np.linspace(X[:,1].min(), X[:,1].max(), 400)
xx1, xx2 = np.meshgrid(x1, x2)
grid = np.c_[xx1.ravel(), xx2.ravel()]

Z = clf.predict(grid).reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.3)
plt.scatter(X[:10000,0], X[:10000,1], c=y[:10000], s=1)
plt.xlabel("Altitude")
plt.ylabel("Aspect")
plt.show()


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


# decision stump on feature 1 by information gain maximisation
n = 10000
inds = np.arange(0, n)
xx1 = np.sort(X[inds,0])
yy = y[inds]
y_bar = np.mean(yy)
gini_full = 1 - y_bar**2 - (1 - y_bar)**2 # around 0.47

midpoints = 0.5 * (xx1[1:] + xx1[:-1])
gini_split = []
for threshold in midpoints:
  mask = (xx1 <= threshold)
  y_left, y_right = yy[mask], yy[~mask]
  y_bar_left, y_bar_right = np.mean(y_left), np.mean(y_right)
  n_left, n_right = len(y_left), len(y_right)
  gini_left = 1 - y_bar_left**2 - (1 - y_bar_left)**2
  gini_right = 1 - y_bar_right**2 - (1 - y_bar_right)**2
  gini_split.append(n_left/n * gini_left + n_right/n * gini_right)

information_gain = gini_full - gini_split
threshold = midpoints[ information_gain == np.max(information_gain) ][0]

# plot target vs feature 1, vertical jitter for better visibility
jitter = np.random.uniform(-.05, .05, size=len(yy))
plt.figure(figsize=(7,3))
plt.plot(xx1, yy + jitter, 'ok', markersize=.5)
plt.tight_layout()
plt.yticks([0,1])
plt.xlabel('feature 1')
plt.ylabel('target')
plt.tight_layout()
plt.show()

# plot information gain vs threshold and indicate threshold choice that
# maximises IG
plt.figure(figsize=(7,4))
plt.plot(midpoints, gini_full - gini_split, '-k')
plt.axvline(x=threshold, ymin=0, ymax=1)
plt.xlabel('feature 1 threshold')
plt.ylabel('information gain')
plt.show()


# plot histogram of targets before splitting
fig, ax = plt.subplots(1,1, figsize=(7,3))
ax.hist(yy, bins=[-.5,.5,1.5], rwidth=.9)
ax.set_xticks([0,1])
ax.set_xlabel('target')
ax.set_ylabel('frequency')
ax.set_title('all data')
plt.tight_layout()
plt.show()


# plot histograms of targets after splitting
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7,4))
ax0.hist(yy[ xx1 <= threshold ], bins=[-.5, .5, 1.5], rwidth=.9)
ax0.set_xticks([0,1])
ax0.set_ylim([0, 4500])
ax0.set_xlabel('target')
ax0.set_ylabel('frequency')
ax0.set_title('x1 <= threshold')
ax1.hist(yy[ xx1 > threshold ], bins=[-.5, .5, 1.5], rwidth=.9)
ax1.set_xticks([0,1])
ax1.set_ylim([0, 4500])
ax1.set_xlabel('target')
ax1.set_title('x1 > threshold')
plt.tight_layout()
plt.show()


# fit random forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# prediction surface over x1/x2 plane
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 400)
x2 = np.linspace(X[:,1].min(), X[:,1].max(), 400)
xx1, xx2 = np.meshgrid(x1, x2)
grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = model.predict(grid).reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3)
# plt.scatter(X[:10000,0], X[:10000,1], c=y[:10000], s=1)
plt.xlabel("Altitude")
plt.ylabel("Aspect")
plt.show()


# probability surface over x1/x2 plane
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 400)
x2 = np.linspace(X[:,1].min(), X[:,1].max(), 400)
xx1, xx2 = np.meshgrid(x1, x2)
grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = model.predict_proba(grid)[:,1].reshape(xx1.shape)
cf = plt.contourf(xx1, xx2, Z, levels=np.linspace(0, 1, 21), vmin=0, vmax=1,)
plt.colorbar(cf, label="probability")
# plt.scatter(X[:10000,0], X[:10000,1], c=y[:10000], s=1)
plt.xlabel("Altitude")
plt.ylabel("Aspect")
plt.show()


