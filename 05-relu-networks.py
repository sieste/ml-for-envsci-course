import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

x = np.linspace(-2, 4, 600)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# 1. Absolute value
f1 = relu(x) + relu(-x)
axes[0].plot(x, f1, label="ReLU(x) + ReLU(-x)")
axes[0].set_title("f(x) = |x|")
axes[0].legend()

# 2. Two-kink PL function using only ReLU units
#    x = ReLU(x) - ReLU(-x)
f2 = (relu(x) - relu(-x)) + 2*relu(x+1) - 3*relu(x-1)
axes[1].plot(x, f2, label="ReLU(x)-ReLU(-x) + 2ReLU(x+1) - 3ReLU(x-1)")
axes[1].set_title("Two-kink piecewise-linear function")
axes[1].legend()

# 3. Full triangle wave (down–up cycle)
f3 = relu(x) - 2*relu(x-1.5) + relu(x-3)
axes[2].plot(x, f3, label="ReLU(x) - 2ReLU(x-1.5) + ReLU(x-3)")
axes[2].set_title("Triangle wave cycle")
axes[2].legend()

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random



def build_model(n_hidden):
    model = keras.Sequential([
        layers.Dense(n_hidden, activation="relu", input_shape=(1,),
          kernel_initializer=keras.initializers.RandomNormal(stddev=3.0),
          bias_initializer=keras.initializers.RandomUniform(-10, 10)),
        layers.Dense(1, activation=None),
    ])
    return model

random.seed(35) 
np.random.seed(0)
tf.random.set_seed(0)
models = [build_model(n) for n in [2, 8, 64]]
x_in = np.linspace(-3, 3, 400).reshape(-1,1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, model in zip(axes, models):
    y = model(x_in).numpy().squeeze()
    ax.plot(x_in.squeeze(), y)
    ax.set_title(f"{model.layers[0].units} ReLU units")
plt.tight_layout()
plt.savefig("fig/nn-randomrelu1d.png", dpi=300, bbox_inches="tight")
plt.show()


# 2d 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 2-D grid
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
XY = np.stack([X, Y], axis=-1).reshape(-1, 2)

def build_model(n_hidden):
    return keras.Sequential([
        layers.Dense(
            n_hidden, activation="relu", input_shape=(2,),
            kernel_initializer=keras.initializers.RandomNormal(stddev=1.0),
            bias_initializer=keras.initializers.RandomUniform(-3, 3),
        ),
        layers.Dense(1, activation=None)
    ])
random.seed(47) 
np.random.seed(0)
tf.random.set_seed(0)
models = [build_model(n) for n in [4, 8, 128]]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, model in zip(axes, models):
    Z = model(XY).numpy().reshape(200, 200)
    im = ax.imshow(Z, extent=(-3, 3, -3, 3), origin="lower")
    ax.contour(X, Y, Z, colors="black", linewidths=0.5)
    ax.set_title(f"{model.layers[0].units} ReLU units")
plt.tight_layout()
plt.savefig("fig/nn-randomrelu2d.png", dpi=300, bbox_inches="tight")
plt.show()


# activation function
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 400)

relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
swish = x * sigmoid
fig, axes = plt.subplots(2, 2, figsize=(4, 3))
axes[0, 0].plot(x, relu, '-k')
axes[0, 0].set_title("ReLU")
axes[0, 1].plot(x, swish, '-r')
axes[0, 1].set_title("Swish")
axes[1, 0].plot(x, sigmoid, '-g')
axes[1, 0].set_title("Sigmoid")
axes[1, 1].plot(x, tanh, '-b')
axes[1, 1].set_title("Tanh")
plt.tight_layout()
plt.savefig("fig/nn-activations.png", dpi=300, bbox_inches="tight")
plt.show()

