from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Dense
import urllib.request
import zipfile

# load dataset
url = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
data_dir = "data"
zip_path = os.path.join(data_dir, "air_quality.zip")
csv_path = os.path.join(data_dir, "AirQualityUCI.csv")
url = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"

if not os.path.exists(csv_path):
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)



# load data
data = np.genfromtxt(
    csv_path,
    delimiter=";",
    skip_header=1,
    dtype=str
)

####################
# columns:
# 0 Date
# 1 Time
# 2 CO(GT)
# 3 PT08.S1(CO)
# 4 NMHC(GT)
# 5 C6H6(GT)
# 6 PT08.S2(NMHC)
# 7 NOx(GT)
# 8 PT08.S3(NOx)
# 9 NO2(GT)
# 10 PT08.S4(NO2)
# 11 PT08.S5(O3)
# 12 T
# 13 RH
# 14 AH
####################

data.shape # (9471, 17) - the last 2 columns are empty

data[0,:] # complete row
data[5000, :] # a row with missing values

# replace "-200" by "" to indicate missing value
data[(data == '-200') | (data == '-200,0')] = ''

# remove rows that are all empty
data = data[~np.all(data == '', axis=1)] 

# store date time
dt = np.array([
    datetime.strptime(d + " " + t, "%d/%m/%Y %H.%M.%S")
    for d, t in zip(data[:, 0], data[:, 1])
])

# decimals are stored in format "11,9", need to be converted to 11.9 
# NOTE: this step deletes the dates and times
def to_float(x):
  try:
    return float(x.replace(",", "."))
  except:
    return np.nan
to_float = np.vectorize(to_float) # can now accept array inputs
data = to_float(data)


# extract target and features
i_target = [2]
i_sensor = [3,6,8,10,11]
i_meteo = [12,13,14]
i_xy = i_target + i_sensor + i_meteo


# show some data
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(dt, data[:, i_target], "-r")
ax[0].set_title("CO ground truth")
ax[1].plot(dt, data[:, i_sensor[0]], "-b")
ax[1].set_title("CO sensor reading")
ax[2].plot(dt, data[:, i_meteo[1]], "-g")
ax[2].set_title("Rel. Humidity")
fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()


# handle missing data (either nan or -200)
na_rows = np.any(np.isnan(data[:,i_xy]), axis=1)
data = data[~na_rows]
dt = dt[~na_rows]

# construct targets and features
y = data[:, i_target]
X = data[:, i_sensor + i_meteo]

# train / test split
X_train, X_test, y_train, y_test = \
  train_test_split(X, y, test_size=0.2, random_state=42)

# scale each feature by (x - mean(x))/sd(x) 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define MLP model
model = keras.Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])


# compile
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100, batch_size=32,
    verbose=1)

# learning curve
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("epoch")
plt.ylabel("mse")
plt.legend()
plt.show()

# evaluate
yhat_test = model(X_test)
yhat_clim = np.mean(y_train) * np.ones_like(y_test)

# compare MSE and MAE to climatology
mse = np.mean(np.square(y_test - yhat_test))
mse_clim = np.mean(np.square(y_test - yhat_clim))

print(f"Test MSE: {mse}\nClim MSE: {mse_clim}")



# comparison with multiple linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
linreg = LinearRegression()
linreg.fit(X_train, y_train)
yhat_lr = linreg.predict(X_test)
mse_test_lr = np.mean(np.square(yhat_lr - y_test))
print(f"MLP MSE: {mse}\nLR MSE: {mse_test_lr}")


# plot some model outputs
X_full = scaler.transform(X)
yhat_full = model(X_full)
yhat_lr_full = linreg.predict(X_full)

inds = 50
plt.plot(dt[:inds], y[:inds,0], label='target')
plt.plot(dt[:inds], yhat_full[:inds,0], label='MLP')
plt.plot(dt[:inds], yhat_lr_full[:inds,0], label='Linear regression')
plt.legend()
plt.tight_layout()
plt.ylabel('CO concentration')
fig = plt.gcf()
fig.autofmt_xdate(rotation=45)
plt.show()

ax[0].set_title("CO ground truth")
ax[1].plot(dt, data[:, i_sensor[0]], "-b")
ax[1].set_title("CO sensor reading")
ax[2].plot(dt, data[:, i_meteo[1]], "-g")
ax[2].set_title("Rel. Humidity")
plt.tight_layout()
plt.show()




# next steps: impute missing values with (training) mean and provide a
# missingness mask as an input to the NN



