

# Python setup with uv on macOS

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

Restart your terminal, then check:

```bash
uv --version
```

## 2. Create a project

```bash
mkdir my-python-project
cd my-python-project
uv init
```

## 3. Install packages

```bash
uv add torch matplotlib scikit-learn numpy pandas seaborn
```

## 4. Run Python code

Create a file:

```bash
touch main.py
```

Example `main.py`:

```python
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

print("Torch version:", torch.__version__)

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Sample classification data")
plt.show()
```

Run it:

```bash
uv run python main.py
```
It should print something like this
```
Torch version: 2.11.0
```
