import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import random

# Define image size
IMG_SIZE = (128, 128)

# Path to dataset
DATASET_PATH = "/Users/Allana/Dev/pytorch practice/Brain tumor images/"
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

# Function to load images and labels
def load_data(folder):
    data, labels = [], []
    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, folder, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, IMG_SIZE)  # Resize
            data.append(img.flatten())  # Flatten into 1D feature vector
            labels.append(category)
    return np.array(data), np.array(labels)

# Load training and test data
X_train_subset, y_train_subset = load_data("Testing") #IF SUBSET CHANGE BACK TO FULL
X_test, y_test = load_data("Training")

# subset_indices = random.sample(range(len(X_train_full)), 1300) 
# X_train_subset = X_train_full[subset_indices]
# y_train_subset = y_train_full[subset_indices]




# Encode labels into numbers
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train_subset)
y_test_enc = encoder.transform(y_test)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_subset)
X_test = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train_enc)

# Evaluate model
y_pred = svm_model.predict(X_test)
print(classification_report(y_test_enc, y_pred, target_names=CATEGORIES))
