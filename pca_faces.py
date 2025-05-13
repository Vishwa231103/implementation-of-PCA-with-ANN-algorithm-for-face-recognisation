import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming X and y are loaded from your previous code
# If you are starting from a new file, re-import X and y here

from load_faces import X, y  # This line assumes both scripts are in the same folder

# Step 1: Normalize (optional but useful)
X = X / 255.0  # Scale pixel values between 0 and 1

# Step 2: Apply PCA
pca = PCA(n_components=100)  # Try reducing to 100 dimensions
X_pca = pca.fit_transform(X)

print("PCA applied successfully!")
print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)

# Step 3: Optional - Plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()
