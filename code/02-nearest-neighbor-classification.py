import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# in ipython this line ensures the terminal isn't blocked by plotting commands,
# might not be necessary in other IDEs
plt.ion()

# # download temperature data (skip if file already exists)
# import urllib.request
# import os
# url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
# target = "data/penguins.csv"
# if not os.path.isfile(target):
#   urllib.request.urlretrieve(url, target)

# columns:
# species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year

# load data 
data_file = "data/penguins.csv"
penguins = np.loadtxt(data_file, skiprows=1, dtype='str', delimiter=",")

# remove NAs
na_mask = np.any(penguins == 'NA', axis=1)
penguins = penguins[~na_mask]

# two features (bill_length_mm, bill_depth_mm)
# X = np.array([(x[2], x[3]) for x in penguins], dtype=np.float32)
X = np.float32(penguins[:, 2:4])

# one target (sex)
# y = np.array([x[6] for x in penguins])
y = penguins[:, 6]

# plot x1/x2 features, color by sex
males = (y == 'male')
plt.plot(X[males, 0], X[males, 1], 'o', color='blue', label='male')
females = (y == 'female')
plt.plot(X[females, 0], X[females, 1], 'o', color='orange', label='female')
plt.legend()
plt.xlabel('bill length')
plt.ylabel('bill depth')
plt.show()

# x1 and x2 have quite different scales (x1 = 34...60, x2 = 13...21) since we
# will define neighborhood by euclidean distances, normalisation to a common
# range makes sense, here we use 0-1 normalisation
X[:, 0] = (X[:, 0] - np.min(X[:, 0])) / (np.max(X[:,0]) - np.min(X[:,0]))
X[:, 1] = (X[:, 1] - np.min(X[:, 1])) / (np.max(X[:,1]) - np.min(X[:,1]))

# initialise nearest neighbor classifier 
clf = KNeighborsClassifier(n_neighbors=5)

# train the classifier
clf.fit(X, y)

# predict labels for training features
y_hat = clf.predict(X)

# plot the classification surface over a dense grid of x0,x1 values
x0, x1 = np.mgrid[0:1:100j, 0:1:100j]
x_grid = np.column_stack([x0.ravel(), x1.ravel()])
y_hat = clf.predict(x_grid)
males, females = (y_hat == 'male'), (y_hat == 'female')
plt.plot(x_grid[males, 0], x_grid[males, 1], '.', color='blue', 
         markersize=5, alpha=.3)
plt.plot(x_grid[females, 0], x_grid[females, 1], '.', color='orange', 
         markersize=5, alpha=.3)
males, females = (y == 'male'), (y == 'female')
plt.plot(X[males, 0], X[males, 1], 'o', color='blue', label='male')
plt.plot(X[females, 0], X[females, 1], 'o', color='orange', label='female')
plt.legend()
plt.xlabel('bill length (normalised)')
plt.ylabel('bill depth (normalised)')
plt.title("k = 5")
plt.show()


# explore how decision surface depends on neighborhood size
k_vals = [1,2,3,4,5,10,15,20]
fig, axs = plt.subplots(1, len(k_vals), figsize=(10,1.8))
for i,k in enumerate(k_vals):
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(X, y)
  y_hat = clf.predict(x_grid)
  males, females = (y_hat == 'male'), (y_hat == 'female')
  axs[i].plot(x_grid[males, 0], x_grid[males, 1], '.', color='blue', markersize=1)
  axs[i].plot(x_grid[females, 0], x_grid[females, 1], '.', color='orange', markersize=1)
  axs[i].set_xticks([])
  axs[i].set_yticks([])
  axs[i].set_title(f"k = {k}")
plt.tight_layout()
plt.show()


# split data into training data used to train the classifier and a test data
# used to evaluate the classifier
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.75,
    random_state=0, shuffle=True)


# loop over a range of k values
k_vals = np.arange(1, 21)
loss_tr, loss_te = [], []
for k in k_vals:
  # train knn classifier on training data
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(X_tr, y_tr)
  # calculate training accuracy and test loss
  y_hat_tr = clf.predict(X_tr)
  y_hat_te = clf.predict(X_te)
  loss_tr.append(np.mean(y_hat_tr == y_tr))
  loss_te.append(np.mean(y_hat_te == y_te))


# plot training and test accuracy vs k
plt.plot(k_vals, loss_tr, label='training loss')
plt.plot(k_vals, loss_te, label='test loss')
plt.xticks(k_vals)
plt.xlabel('k')
plt.ylabel('0-1 loss (accuracy)')
plt.legend()
plt.show()


