import os
import urllib.request
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# download temperature data (skip if file already exists)
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
target = "data/penguins.csv"
if not os.path.isfile(target):
  urllib.request.urlretrieve(url, target)

# columns:
# species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year

# load data (example assumes CSV available locally or via URL)
penguins = np.loadtxt(target, skiprows=1, dtype='str', delimiter=",")

# remove NAs
penguins = [row for row in penguins if not np.any(row == 'NA')]

# two quantitative features:
x = np.array([(x[2], x[3]) for x in penguins], dtype=np.float32)
y = np.array([x[6] for x in penguins])

# plot
for sex, col in zip(['male','female'], ['blue','orange']):
  mask = (y == sex)
  plt.plot(x[mask, 0], x[mask, 1], 'o', color=col, label=sex)
plt.legend()
plt.xlabel('bill length')
plt.ylabel('bill depth')
plt.show()

# x1 and x2 have quite different scales (x1 = 34...60, x2 = 13...21) since we
# will define neighborhood by euclidean distances, normalisation to a common
# range makes sense, here we use 0-1 normalisation
for i in [0,1]:
  x[:, i] = (x[:, i] - np.min(x[:, i])) / (np.max(x[:,i]) - np.min(x[:,i]))

# fit nearest neighbor classifier with 
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x, y)
y_hat = clf.predict(x)

# plot classification surface
x_vals = np.linspace(0, 1, 100)
x_grid = np.array([(x0, x1) for x0 in x_vals for x1 in x_vals])
y_hat = clf.predict(x_grid)
for sex, col in zip(['male','female'], ['blue','orange']):
  mask = (y_hat == sex)
  plt.plot(x_grid[mask, 0], x_grid[mask, 1], '.', color=col, markersize=5, alpha=.3)
for sex, col in zip(['male','female'], ['blue','orange']):
  mask = (y == sex)
  plt.plot(x[mask, 0], x[mask, 1], 'o', color=col, label=sex)
plt.legend()
plt.xlabel('bill length (normalised)')
plt.ylabel('bill depth (normalised)')
plt.title("k = 5")
plt.show()


k_vals = [1,2,3,4,5,10,15,20]
fig, axs = plt.subplots(1, len(k_vals), figsize=(10,1.8))
for i,k in enumerate(k_vals):
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(x, y)
  y_hat = clf.predict(x_grid)
  for sex, col in zip(['male','female'], ['blue','orange']):
    mask = (y_hat == sex)
    axs[i].plot(x_grid[mask, 0], x_grid[mask, 1], '.', color=col, markersize=1)
  axs[i].set_xticks([])
  axs[i].set_yticks([])
  axs[i].set_title(f"k = {k}")
plt.tight_layout()
plt.show()



# split data into training data used to train the classifier and a test data
# used to evaluate the classifier
np.random.seed(0)
n = len(x)
i_train = np.random.choice(len(x), 3 * n // 4, replace=False)
i_train = np.random.choice(len(x), n // 2, replace=False)
i_test = np.setdiff1d(list(range(n)), i_train)
x_tr, y_tr = x[i_train], y[i_train]
x_te, y_te = x[i_test], y[i_test]

# loop over a range of k values
k_vals = np.arange(1, 21)
loss_tr, loss_te = [], []
for k in k_vals:
  # train knn classifier on training data
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(x_tr, y_tr)
  # calculate training loss and test loss
  y_hat_tr = clf.predict(x_tr)
  y_hat_te = clf.predict(x_te)
  loss_tr.append(np.mean(y_hat_tr == y_tr))
  loss_te.append(np.mean(y_hat_te == y_te))

# plot training and test loss over k
plt.plot(k_vals, loss_tr, label='training loss')
plt.plot(k_vals, loss_te, label='test loss')
plt.xticks(k_vals)
plt.xlabel('k')
plt.ylabel('0-1 loss')
plt.legend()
plt.show()

