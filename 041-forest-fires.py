from pathlib import Path
import requests
import csv

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# download bejaia forest fires file into data/ if it doesn't exist already,
# throw error if url cannot be reached
url = "https://raw.githubusercontent.com/jsimkins2/geog473-673/refs/heads/master/datasets/Bejaia_ForestFires.csv"
dest = Path("data/Bejaia_ForestFires.csv")
if not dest.exists():
  r = requests.get(url, timeout=30)
  r.raise_for_status()
  dest.write_bytes(r.content)

# open data file
with open("data/Bejaia_ForestFires.csv", newline="", encoding="utf-8") as f:
  reader = csv.DictReader(f)
  rows = list(reader)

# Date: Day, month (‘june’ to ‘september’) for the year 2012 
# Temp: Max daily temperature in degrees Celsius: 22 to 42 
# RH: Relative Humidity in %: 21 to 90 
# Ws: Wind speed in km/h: 6 to 29 
# Rain: Total daily precip in mm: 0 to 16.8 
# Fire Weather Indices:
# FFMC: Fine Fuel Moisture Code index from the FWI system: 28.6 to 92.5 
# DMC: Duff Moisture Code index from the FWI system: 1.1 to 65.9 
# DC: Drought Code index from the FWI system: 7 to 220.4 
# ISI: Initial Spread Index from the FWI system: 0 to 18.5 
# BUI: Buildup Index from the FWI system: 1.1 to 68 
# FWI: Class Weather Index : 0 to 31.1 
# Class : Forest Fire presence (fire) or absence (no fire)

# here we use temperature, relative humidity, wind speed and rain as features
X = np.loadtxt("data/Bejaia_ForestFires.csv", delimiter=",", usecols=(2,3,4,5), encoding="utf-8", skiprows=1)

# to predict the binary label of Class: fire/no fire
y = np.loadtxt("data/Bejaia_ForestFires.csv", delimiter=",", dtype=np.str_, usecols=(12), skiprows=1)

# encode fire/no fire labels to 1/0
le = LabelEncoder()
y_enc = le.fit_transform(y)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

# with only about 100 examples we should keep the model fairly simple to avoid
# overfitting
clf = DecisionTreeClassifier(
    max_depth=4, min_samples_split=5, min_samples_leaf=2, max_features=2, 
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# show fitted tree
tree.plot_tree(clf, feature_names=['temp','rh','wind','rain'], 
               class_names=['fire','no fire'], fontsize=8)
plt.show()

# evaluate fitted tree
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, normalize='true')
report = classification_report(y_test, y_pred)





