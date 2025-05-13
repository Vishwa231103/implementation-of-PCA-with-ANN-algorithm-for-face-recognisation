import joblib

# Load saved models
pca = joblib.load('pca_model.pkl')
model = joblib.load('face_recognition_model.pkl')
