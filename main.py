import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

print("Torch version:", torch.__version__)

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Sample classification data")
plt.show()