import cv2
import numpy as np

# Load image (Update this path to your test image!)
new_image_path = r'C:\Users\vishw\OneDrive\Desktop\internship project\dataset\faces\Aamir\face_5.jpg'
new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

if new_img is None:
    print("❌ Failed to load image. Please check the file path.")
    exit()

# Resize and flatten
new_img_resized = cv2.resize(new_img, (100, 100))
new_img_flattened = new_img_resized.flatten().reshape(1, -1)

# Apply PCA and predict
new_img_pca = pca.transform(new_img_flattened)
predicted_label = model.predict(new_img_pca)

print("✅ Predicted label (person ID):", predicted_label[0])
