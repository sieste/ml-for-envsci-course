import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# # The code below generates the data used in the course lecture. You can
# # adapt the code to create your own training data.
# #
# # data source https://www.kaggle.com/datasets/alessiocorrado99/animals10
# # downloaded archive contains directory "raw-img" and file "translate.py"
# # 
# # moved/renamed the directory to data/animals-10-raw-img (base_dir below)
# # 
# # translate.py contains the following dictionary:
# translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
# base_dir = "data/animals-10-raw-img"
# classes_en = ["chicken", "elephant", "butterfly"]
# classes = [translate[x] for x in classes_en]
# n_per_class = 500
# img_size = 100 # width and height in pixels
# images = np.zeros((len(classes) * n_per_class, img_size, img_size, 3), dtype=np.uint8)
# labels = np.empty(len(classes) * n_per_class, dtype=object)
# idx = 0
# for cls in classes:
#   cls_dir = os.path.join(base_dir, cls)
#   files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpeg")]
#   selected = files[:n_per_class]
#   for fname in selected:
#     path = os.path.join(cls_dir, fname)
#     img = Image.open(path).convert("RGB")
#     w, h = img.size
#     # crop image into square format
#     if w > h:
#       left = (w - h) // 2
#       img = img.crop((left, 0, left + h, h))
#     elif h > w:
#       top = (h - w) // 2
#       img = img.crop((0, top, w, top + w))
#     # resize image
#     img = img.resize((img_size, img_size), Image.BILINEAR)
#     images[idx] = np.array(img)
#     labels[idx] = translate[cls]
#     idx += 1
# np.save('data/animal-images.npy', images)
# np.save('data/animal-labels.npy', labels)

# load the stored data
images = np.load('data/animal-images.npy', allow_pickle=True)
labels = np.load('data/animal-labels.npy', allow_pickle=True)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# images: (1200, 50, 50, 3), labels: (1200,)
X = images.astype("float32") / 255.0  # min-max scaling

# one-hot encoding of class labels
le = LabelEncoder()
y_int = le.fit_transform(labels) # (n_samples,)
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y_int.reshape(-1,1))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_int)

model = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(y.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)


# pick random test sample
i = np.random.randint(len(X_test))
img = X_test[i]
true_idx = np.argmax(y_test[i])
true_label = le.classes_[true_idx]
probs = model.predict(img[np.newaxis, ...])[0]
# plot image
plt.imshow(img)
plt.axis("off")
# build caption
lines = []
for j, cls in enumerate(le.classes_):
    line = f"{cls}: {probs[j]:.3f}"
    if j == true_idx:
        line += "  <-- correct"
    else:
        line += "             "
    lines.append(line)
plt.title("\n".join(lines), fontsize=9, fontfamily="monospace")
plt.show()

# calculate class probabilities and majority prediction 
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)
disp.plot(cmap="Blues", values_format="d")
plt.show()

# accuracy score (random guessing has accuracy 1/n_classes)
acc_test = np.mean(y_pred == y_true)

# log loss of model probabilities
from sklearn.metrics import log_loss
log_loss(y_true, y_pred_prob), log_loss(y_true, 1/3 * np.ones_like(y_pred_prob))
# interpretation: log-loss difference of 0.5 means the trained model assigns
# exp(.5) = 1.65 more probability to the true class than the baseline



# next steps:
# build a cats vs dogs classifier from the same data set
# explore architecture and hyperparameters
# hyperparameter optimisation
# more validation metrics for classifiers (f1, roc, calibration curve)

