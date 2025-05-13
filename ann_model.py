import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ✅ Step 1: Define these based on your PCA and dataset
desired_pca_dimension = 50    # Set this to the number of PCA components you used
num_classes = 10              # Set this to the number of unique individuals (classes)

# ✅ Step 2: Initialize the model
model = Sequential()

# Input layer
model.add(Dense(128, input_dim=desired_pca_dimension, activation='relu'))

# Hidden layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# ✅ Step 3: Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ Step 4: Summarize the model structure
model.summary()
