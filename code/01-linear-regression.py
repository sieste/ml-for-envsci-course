import os
import numpy as np
from matplotlib import pyplot as plt

# in ipython this line ensures the terminal isn't blocked by plotting commands,
# might not be necessary in other IDEs
plt.ion()

# temperature data from https://climexp.knmi.nl/gettemp.cgi?WMO=7110
data_file = "data/t7110.dat"

# load file with numpy
t7110 = np.loadtxt(data_file, comments="#")

# extract features (july temperatures) and targets (august temperatures)
x = t7110[:, 7]
y = t7110[:, 8]

# remove any instances with missing values
na_mask = (x == -999.9) | (y == -999.9)
x = x[~na_mask]
y = y[~na_mask]

# linear regression with sklearn
from sklearn.linear_model import LinearRegression

# sklearn expects k features as (n, k) array, here k=1
X = x.reshape(-1, 1)

# initialise model object
model = LinearRegression()

# fit (train) the model
model.fit(X, y)

# extract parameter estimates
theta0_skl, theta1_skl = model.intercept_, model.coef_[0]
(theta0_skl, theta1_skl)

# calculate predictions 
y_hat = model.predict(X)


plt.plot(x, y, "ob")
plt.plot(x, y_hat, "-r")
plt.show()


# plot x/y scatter plot and linear regression line
X_new = np.linspace(10,20,100).reshape(-1,1)
y_new = model.predict(X_new)
plt.plot(X_new, y_new, '-r', label='LR fit')
plt.plot(x,y,'.k', label='data')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(y), np.max(y)])
plt.xlabel('July temperature')
plt.ylabel('August temperature')
plt.legend()
plt.show()


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


