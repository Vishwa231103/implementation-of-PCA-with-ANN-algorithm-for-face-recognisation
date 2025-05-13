import cv2
import numpy as np
import joblib

# Load models
pca = joblib.load('pca_model.pkl')
model = joblib.load('face_recognition_model.pkl')

# Load new image
new_image_path = r'C:\Users\vishw\OneDrive\Desktop\internship project\dataset\faces\Aamir\faces.jpg'
new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

if new_img is None:
    print("❌ Failed to load image. Please check the file path.")
    exit()

# Preprocess
new_img_resized = cv2.resize(new_img, (100, 100))
new_img_flattened = new_img_resized.flatten().reshape(1, -1)

# Apply PCA
new_img_pca = pca.transform(new_img_flattened)

# Predict
predicted_label = model.predict(new_img_pca)
print("✅ Predicted label (person ID):", predicted_label[0])
