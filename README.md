# ML from Scratch 🧑‍💻

A collection of classic machine learning algorithms implemented from scratch in Python.  
The goal is educational: understand how algorithms work under the hood by building them step by step.

## 📂 Project Structure
- **`src/mlscratch/`** → Core implementations of algorithms  
- **`examples/`** → Jupyter notebooks with usage demos  
- **`datasets/`** → Small datasets (CSV or loaders)  



## ✅ Implemented Algorithms
- Linear regression
  -   Notes: Standardizing data (X, y) makes a big difference for the gradient-based method. Fix the overflow problem.
  -   [Full derivation of loss & gradients](docs/linear_regression_math.md)
  
- Logistic Regression
  -  [Full derivation of loss & gradients](docs/logistic_regression_math.md)

- Softmax Regression
  -  [Full derivation of loss & gradients](docs/softmax_regression_math.md)

- Decision Trees, Random Forest
  -  [Full derivation of loss & gradients](docs/decision_tree_math.md)
- KNN
- Naive Bayes
- SVM (linear + kernel)
- PCA, LDA
- KMeans, GMM
- Neural Networks (simple MLP)

## 🚀 Getting Started
```bash
git clone https://github.com/x35yao/ml-from-scratch.git
cd ml-from-scratch
pip install -e .[dev]







