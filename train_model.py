import joblib

joblib.dump(pca, 'pca_model.pkl')
joblib.dump(model, 'face_recognition_model.pkl')
print("✅ PCA and model saved!")
