import os
import random
import numpy as np
from PIL import Image

translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
base_dir = "data/animals-10-raw-img"
classes_en = ["chicken", "elephant", "butterfly"]
classes = [translate[x] for x in classes_en]
n_per_class = 1000
img_size = 100

images = np.zeros((len(classes) * n_per_class, img_size, img_size, 3), dtype=np.uint8)
labels = np.empty(len(classes) * n_per_class, dtype=object)

idx = 0
for cls in classes:
    cls_dir = os.path.join(base_dir, cls)
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpeg")]
    selected = files[:n_per_class]
    for fname in selected:
        path = os.path.join(cls_dir, fname)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > h:
            left = (w - h) // 2
            img = img.crop((left, 0, left + h, h))
        elif h > w:
            top = (h - w) // 2
            img = img.crop((0, top, w, top + w))
        img = img.resize((img_size, img_size), Image.BILINEAR)
        images[idx] = np.array(img)
        labels[idx] = translate[cls]
        idx += 1



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# images: (1200, 50, 50, 3), labels: (1200,)
X = images.astype("float32") / 255.0  # min-max scaling

le = LabelEncoder()
y_int = le.fit_transform(labels)
y = to_categorical(y_int)  # one-hot encoding

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_int
)

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
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

test_loss, test_acc = model.evaluate(X_test, y_test)


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



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# show confusion matrix
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)
disp.plot(cmap="Blues", values_format="d")
plt.show()



# next steps:
# cats vs dogs classifier
# explore architecture and hyperparameters
# hyperparameter optimisation


