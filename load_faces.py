import os
import cv2
import numpy as np

data_path = 'dataset/faces'  # Point to your 'faces' folder
labels = []
images = []

label_num = 0

# Loop through actor folders
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        for image_file in os.listdir(folder_path):
            if image_file.endswith('.pgm') or image_file.endswith('.jpg') or image_file.endswith('.png'):
                img_path = os.path.join(folder_path, image_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (100, 100))  # Resize to 100x100
                images.append(img_resized.flatten())  # Flatten to 1D
                labels.append(label_num)
        label_num += 1

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

print("Face data loaded successfully!")
print("Total samples:", len(X))
print("Each sample shape (flattened):", X[0].shape)
print("Total unique labels:", len(set(y)))
