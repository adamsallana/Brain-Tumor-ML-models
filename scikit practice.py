import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "path_to_your_dataset"

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

# load data
image_data = []
labels = []

# Step 1: Load and preprocess images
for category in CATEGORIES:
    folder_path = os.path.join(DATASET_PATH, category)
    
    # Loop over each image in the folder
    for image_name in os.listdir(folder_path):
        if image_name.endswith(".jpg"):
            img_path = os.path.join(folder_path, image_name)
            
            # Step 2: Read and resize image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale)
            image_resized = cv2.resize(image, (128, 128)) 
            
            # Step 3: Flatten the image to create a feature vector
            image_data.append(image_resized.flatten())  # Flatten the 2D image to 1D vector
            
            # Step 4: Assign label (0 for glioma, 1 for meningioma, etc.)
            labels.append(CATEGORIES.index(category))

# Convert to NumPy arrays
X = np.array(image_data)
y = np.array(labels)


# # split into test and train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#set data paths for test and train folder



# normalize vals
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# using svm model
model = SVC(kernel="rbf", C=1.0, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# eval model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))

#Testing w/ image

def predict_image(image_path, model, scaler, label_encoder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img.flatten().reshape(1, -1)
    img = scaler.transform(img)

    prediction = model.predict(img)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    plt.imshow(cv2.imread(image_path), cmap="gray")
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()

    return predicted_label

# sample test
test_image_path = "path_to_sample_image.jpg" #CHANGE PATH
predicted_category = predict_image(test_image_path, model, scaler, label_encoder)
print(f"Prediction: {predicted_category}")

