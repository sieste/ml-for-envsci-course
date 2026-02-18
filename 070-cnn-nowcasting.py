import numpy as np
import random
import zipfile

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input

from matplotlib import pyplot as plt

# Data source: https://zenodo.org/records/17713618 -- Download the file
# "NimrodMLdata-1000x40x40x20-seed1.dat.zip" from there into data/ directory


# unzip into data/
zip_path = "data/NimrodMLdata-1000x40x40x20-seed1.dat.zip"
with zipfile.ZipFile(zip_path) as zf:
    zf.extract("NimrodMLdata-1000x40x40x20-seed1.dat", path="data/")

# read into 4d array
W, H, C, B = 40, 40, 20, 1000
data = np.loadtxt('data/NimrodMLdata-1000x40x40x20-seed1.dat', comments='#', dtype=int)
movies = data.ravel().reshape(B, C, H, W).transpose(0, 2, 3, 1)

# verify shape
movies.shape # (1000, 40, 40, 20)

# plot part of a sequence
i_plt = 40
fig,axs = plt.subplots(1, 5, figsize=(13,4))
plot_arr = movies[i_plt,:,:,:5]
vmax = np.max(plot_arr)
for j in range(5):
  axs[j].imshow(movies[i_plt,:,:,j], vmin=0, vmax=vmax, cmap='Blues')
  axs[j].set_xticks(np.arange(-1, 40, 10))
  axs[j].set_yticks(np.arange(-1, 40, 10))
  axs[j].grid(color='grey', linestyle='--')
  title = f"input {j+1}" if j < 4 else "target"
  axs[j].set_title(title)
  axs[j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.show()


# convert to float32 assumed by keras
movies = movies.astype(np.float32, copy=False)

X = movies[:, :, :, 0:4] # inputs: 4 frames
y = movies[:, :, :, 4:5] # output: 5th frame

# training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialise model
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
model = Sequential([
  Input(shape=(40, 40, 4)),
  Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
  Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
  Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
  Conv2D(filters=1, kernel_size=3, padding='same', activation='relu')
])
# compile and fit model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=32, epochs=100)
# calculate test MSE 
pred_test = model.predict(X_test)
mse_test = np.mean(np.square(pred_test - y_test))
bias_test = np.mean(pred_test - y_test)


# calculate MSE of the persistence prediction
pred_test_ref = X_test[:, :, :, 3:4]
mseref_test = np.mean(np.square(pred_test_ref - y_test))
biasref_test = np.mean(pred_test_ref - y_test)



# plot part of a sequence
i_plt = 40
fig,axs = plt.subplots(1, 5, figsize=(13,4))
plot_arr = movies[i_plt,:,:,:5]
vmax = np.max(plot_arr)
for j in range(5):
  axs[j].imshow(movies[i_plt,:,:,j], vmin=0, vmax=vmax, cmap='Blues')
  axs[j].set_xticks(np.arange(-1, 40, 10))
  axs[j].set_yticks(np.arange(-1, 40, 10))
  axs[j].grid(color='grey', linestyle='--')
  title = f"input {j+1}" if j < 4 else "target"
  axs[j].set_title(title)
  axs[j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.show()

i_plt = 3 # also good: 27
fig,ax = plt.subplots(1,4,figsize=(12,3))
for i in range(4):
  ax[i].imshow(X_test[i_plt,:,:,i], cmap='Blues')
  ax[i].grid()
  ax[i].set_title(f"Input {i+1}")
plt.show()

fig,ax = plt.subplots(1,2,figsize=(12,6))
vmax = np.max([pred_test[i_plt,...], y_test[i_plt,...]])
ax[0].imshow(pred_test[i_plt,...], vmax=vmax, cmap='Blues')
ax[0].grid()
ax[0].set_title("Prediction")
ax[1].imshow(y_test[i_plt,...], vmax=vmax, cmap='Blues')
ax[1].grid()
ax[1].set_title("Target")
plt.show()


################################################################################
################################################################################
################################################################################
################################################################################


# hyperparameters to play with
# - number of conv2d layers
# - numbers of filters
# - kernel size
# - activation

# MORE HERE


# u-net
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
inputs = Input(shape=(40, 40, 4))
c1 = Conv2D(32, 3, padding="same", activation="relu")(inputs)
c1 = Conv2D(32, 3, padding="same", activation="relu")(c1)
p1 = MaxPooling2D()(c1)
c2 = Conv2D(64, 3, padding="same", activation="relu")(p1)
c2 = Conv2D(64, 3, padding="same", activation="relu")(c2)
u1 = UpSampling2D()(c2)
u1 = Concatenate()([u1, c1])
c3 = Conv2D(32, 3, padding="same", activation="relu")(u1)
c3 = Conv2D(32, 3, padding="same", activation="relu")(c3)
outputs = Conv2D(1, 1, padding="same", activation="linear")(c3)
model2 = Model(inputs, outputs)
model2.compile(optimizer = 'adam', loss = 'mse')
model2.fit(X_train, y_train, batch_size=32, epochs=100)
pred2_test = model2.predict(X_test)
bias2_test = np.mean(pred2_test - y_test)
mse2_test = np.mean(np.square(pred2_test - y_test))


print({'mse': (mse_test, mse2_test, mseref_test),
       'bias': (bias_test, bias2_test, biasref_test)})

i_plt = 3 # also good: 27
vmax = np.max([y_test[i_plt,...], pred_test[i_plt,...], pred2_test[i_plt,...]])
fig,ax = plt.subplots(1,3,figsize=(12,3))
ax[0].imshow(pred_test[i_plt,...], vmax=vmax, cmap='Blues')
ax[0].set_title('Deep CNN')
ax[1].imshow(pred4_test[i_plt,...], vmax=vmax, cmap='Blues')
ax[1].set_title('U-Net')
ax[2].imshow(y_test[i_plt,...], vmax=vmax, cmap='Blues')
ax[2].set_title('target')
for i in range(3):
  ax[i].grid()
plt.show()



