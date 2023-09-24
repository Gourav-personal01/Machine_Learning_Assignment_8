# Q6. Using the wine quality data set, perform principal component analysis (PCA) to reduce the number of
# features. What is the minimum number of principal components required to explain 90% of the variance in
# the data?

# Answer - 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the wine quality dataset
wine_data = pd.read_csv('wine_quality_dataset.csv')

# Separate features (X) and target (y)
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Standardize the features (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Plot cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.grid(True)

# Find the number of principal components that explain 90% of the variance
n_components_90_variance = (cumulative_explained_variance >= 0.90).sum()
print(f"Number of Principal Components to explain 90% variance: {n_components_90_variance}")
