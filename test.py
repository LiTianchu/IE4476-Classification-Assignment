# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Load and pre-process the dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to apply dimensionality reduction and classification
def dimensionality_reduction_classification(X_train, X_test, y_train, y_test, method, n_components):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'LDA':
        reducer = LDA(n_components=n_components)
    else:
        raise ValueError("Unsupported method. Choose 'PCA' or 'LDA'.")

    # Apply dimensionality reduction
    X_train_reduced = reducer.fit_transform(X_train, y_train)
    X_test_reduced = reducer.transform(X_test)

    # Apply k-NN classifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train_reduced, y_train)
    y_pred = classifier.predict(X_test_reduced)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Define dimensions to test
dimensions = [10, 20, 50, 100]

# Run experiments for PCA and LDA
results = {'PCA': [], 'LDA': []}
for dim in dimensions:
    print(f"\nEvaluating for {dim} dimensions:")
    pca_accuracy = dimensionality_reduction_classification(X_train, X_test, y_train, y_test, 'PCA', dim)
    lda_accuracy = dimensionality_reduction_classification(X_train, X_test, y_train, y_test, 'LDA', min(dim, len(np.unique(y)) - 1)) # LDA needs n_classes - 1 max components

    results['PCA'].append(pca_accuracy)
    results['LDA'].append(lda_accuracy)

    print(f"PCA Accuracy (n_components={dim}): {pca_accuracy:.4f}")
    print(f"LDA Accuracy (n_components={min(dim, len(np.unique(y)) - 1)}): {lda_accuracy:.4f}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(dimensions, results['PCA'], marker='o', label='PCA')
plt.plot(dimensions, results['LDA'], marker='s', label='LDA')
plt.xlabel('Number of Components')
plt.ylabel('Classification Accuracy')
plt.title('Accuracy vs Number of Components for PCA and LDA')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_components.png',format='png')
plt.show()