import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

DATASET_DIR = "dataset"

X = []
y = []

labels = sorted(
    [folder for folder in os.listdir(DATASET_DIR)
     if os.path.isdir(os.path.join(DATASET_DIR, folder))]
)

print("Found signs:", labels)

for label in labels:
    folder_path = os.path.join(DATASET_DIR, label)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize image
            image = cv2.resize(image, (64, 64))

            # Convert image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Normalize pixel values
            image = image / 255.0

            # Convert image into one-dimensional array
            features = image.flatten()

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total training images:", len(X))

if len(X) == 0:
    print("ERROR: No images found in dataset.")
    exit()

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X, y)

# Save model
with open("sign_language_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training completed successfully!")
print("Saved as: sign_language_model.pkl")