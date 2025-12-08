import urllib.request
import os
import numpy as np
from matplotlib import pyplot as plt

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

# plot x/y scatter plot
plt.plot(x, y, '.k')
plt.show()

# create candidate parameter vectors to optimise over
n_theta = 101
theta0 = np.linspace(0,10,n_theta)
theta1 = np.linspace(-1,2,n_theta)
theta = np.array([(t0, t1) for t0 in theta0 for t1 in theta1])

# plot sample of 1000 candidate models
for _ in range(1000):
  i = np.random.choice(len(theta))
  xx = np.linspace(10,20,100)
  yy = theta[i,0] + theta[i,1] * xx
  plt.plot(xx, yy, '-', color='#bbbbbb')
plt.plot(x,y,'.k')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(y), np.max(y)])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# calculate squared-error loss for each (theta0, theta1) pair
L = np.array([np.sum((y - (th[0] + th[1] * x))**2) for th in theta])

# find value corresponding to minimum loss
i_opt = np.where(L == np.min(L))
theta_hat = theta[i_opt,:].squeeze()

# plot heat map of empirical loss and mark minimal-L point
Z = L.reshape(n_theta, n_theta).T
plt.imshow(Z, origin="lower", aspect="auto",
           extent=[np.min(theta0), np.max(theta0), 
                   np.min(theta1), np.max(theta1)])
plt.plot(theta_hat[0], theta_hat[1], 'or')
plt.xlabel("theta_1")
plt.ylabel("theta_2")
plt.colorbar(label="L")
plt.show()


# plot data, sample of candidate models, and optimised model
xx = np.linspace(10,20,100)
for _ in range(1000):
  i = np.random.choice(len(theta))
  yy = theta[i,0] + theta[i,1] * xx
  plt.plot(xx, yy, '-', color='#bbbbbb')
plt.plot(xx, theta_hat[0] + theta_hat[1] * xx, '-r')
plt.plot(x,y,'.k')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(y), np.max(y)])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# calculate predictions
y_hat = theta_hat[0] + theta_hat[1] * x
np.stack([x,y,y_hat]).T

# doing linear regression properly with sklearn
from sklearn.linear_model import LinearRegression
#
# initialise and fit model
X = x.reshape(-1, 1)   # sklearn expects 2D features
model = LinearRegression()
model.fit(X, y)
#
# extract parameter estimates
theta0_skl, theta1_skl = model.intercept_, model.coef_[0]
(theta0_skl, theta1_skl)
#
# calculate predictions 
y_hat = model.predict(X)



# how "good" is the fitted model?
# calculate rmse
rmse = np.sqrt(np.mean((y - y_hat)**2))
# rmse > 0, so the model is not perfect
# off by less than 1 degree on average
#
# benchmark to simple reference predictions 
#
# climatology: overall mean 
clim = np.mean(y)
rmse_clim = np.sqrt(np.mean((y - clim)**2))
# 
# persistence: last available value
rmse_pers = np.sqrt(np.mean((y - x)**2))
# 
# skill scores (relative improvement against benchmark, 0 = no improvement, 1 =
# perfect)
skill_clim = 1 - rmse / rmse_clim
skill_pers = 1 - rmse / rmse_pers
print(f"skill vs climatology: {skill_clim:.3f}\nskill vs persistence: {skill_pers:.3f}")


