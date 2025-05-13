import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# ✅ Folder path where face images are stored
data_dir ="faces"

image_size = (100, 100)  # You can adjust this size

X = []
y = []
label_dict = {}  # Dictionary to assign numerical labels
label_count = 0

# ✅ Read each subfolder (one per individual)
for person_name in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_name)
    if os.path.isdir(person_path):
        label_dict[person_name] = label_count
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, image_size)
                X.append(image.flatten())  # Flatten the image
                y.append(label_count)
        label_count += 1

# ✅ Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# ✅ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data loaded and split. Training samples:", len(X_train), ", Test samples:", len(X_test))
