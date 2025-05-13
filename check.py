import os

# Check if the model file exists in the project folder
if os.path.exists('project_folder/face_recognition_model.h5'):
    print("Model saved successfully!")
else:
    print("Model saving failed!")
