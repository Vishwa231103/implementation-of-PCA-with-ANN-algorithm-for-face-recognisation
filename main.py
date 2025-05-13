import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

# ====== Step 1: Load Face Dataset ======
print("🔄 Loading face dataset...")
data_path = 'dataset/faces'
labels = []
images = []
label_num = 0

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        for image_file in os.listdir(folder_path):
            if image_file.endswith(('.pgm', '.jpg', '.png')):
                img_path = os.path.join(folder_path, image_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (100, 100))
                images.append(img_resized.flatten())
                labels.append(label_num)
        label_num += 1

X = np.array(images)
y = np.array(labels)
print("✅ Dataset loaded:", X.shape, "Labels:", len(set(y)))

# ====== Step 2: Apply PCA ======
print("🔄 Applying PCA...")
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)
print("✅ PCA applied: New shape:", X_pca.shape)

# ====== Step 3: Train ANN Classifier ======
print("🔄 Training ANN model...")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained. Accuracy on test set:", model.score(X_test, y_test))

# ====== Step 4: Save Models ======
joblib.dump(pca, 'pca_model.pkl')
joblib.dump(model, 'face_recognition_model.pkl')
print("✅ PCA and model saved!")

# ====== Step 5: Predict a New Image ======
print("🔍 Predicting a new image...")
test_img_path = r'C:\Users\vishw\OneDrive\Desktop\internship project\dataset\faces\Aamir/face_5.jpg'  # <- Change path if needed
new_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

if new_img is None:
    print("❌ Failed to load test image. Check the path.")
else:
    new_img_resized = cv2.resize(new_img, (100, 100))
    new_img_flattened = new_img_resized.flatten().reshape(1, -1)
    new_img_pca = pca.transform(new_img_flattened)
    predicted_label = model.predict(new_img_pca)
    print("✅ Predicted Label (Actor ID):", predicted_label[0])
