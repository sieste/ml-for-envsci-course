import urllib.request
import os
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# download temperature data (skip if file already exists)
url = "https://climexp.knmi.nl/data/t7110.dat"
target = "data/t7110.dat"
if not os.path.isfile(target):
  urllib.request.urlretrieve(url, target)

# load file
t7110 = np.loadtxt(target, comments="#")

# extract features (july temperatures) and targets (august temperatures)
x = t7110[:, 7]
y = t7110[:, 8]

# remove missing values
na_mask = (x == -999.9) | (y == -999.9)
x = x[~na_mask]
y = y[~na_mask]



# split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_2d = x_train.reshape(-1,1)
x_test_2d = x_test.reshape(-1,1)

degrees = range(1, 5)
train_err = []
test_err = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    Xtr = poly.fit_transform(x_train_2d)
    Xte = poly.transform(x_test_2d)
    model = LinearRegression()
    model.fit(Xtr, y_train)
    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)
    train_err.append(mean_squared_error(y_train, yhat_tr))
    test_err.append(mean_squared_error(y_test, yhat_te))

# plot training vs test error
plt.figure(figsize=(6,4))
plt.plot(degrees, train_err, label="Training MSE")
plt.plot(degrees, test_err, label="Test MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

# fit three example models: low, medium, high complexity
deg_list = [1, 3, 100]
xx = np.linspace(min(x), max(x), 200).reshape(-1,1)

plt.figure(figsize=(6,4))
plt.scatter(x_train, y_train, s=15, alpha=0.5, label="Training data")

for d in deg_list:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    Xtr = poly.fit_transform(x_train_2d)
    XX = poly.transform(xx)

    model = LinearRegression()
    model.fit(Xtr, y_train)
    yy = model.predict(XX)

    plt.plot(xx, yy, label=f"degree {d}")

plt.xlabel("July temperature")
plt.ylabel("August temperature")
plt.legend()
plt.tight_layout()
plt.show()
