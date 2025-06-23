from sklearn.metrics import pairwise_distances
import numpy as np

X = np.array([[0, 1, 1],
              [0, 0, 1],
              [1, 1, 0]])

# full 3×3 Euclidean distance matrix
D = pairwise_distances(X, metric='euclidean')

# Hamming distance between each sample and the first one
hamm = pairwise_distances(X, [X[0]], metric='manhattan').ravel()

print(D)
print("Hamming vs row0:", hamm)
