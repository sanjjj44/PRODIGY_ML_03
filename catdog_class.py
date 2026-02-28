import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
IMG_SIZE = 64
DATA_DIR = "data"
def load_data():
    X = []
    y = []
    for label, folder in enumerate(["cats", "dogs"]):
        folder_path = os.path.join(DATA_DIR, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()   
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)
X, y = load_data()
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)