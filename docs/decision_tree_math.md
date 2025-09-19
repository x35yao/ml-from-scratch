# 🌳 Decision Tree Classifier — Math Notes (Image Equations)

This note uses **image links** for equations (so they render nicely in Markdown viewers that don't support LaTeX).

---

## 1) Impurity Measures

**Class probabilities at a node**
<p align="center">
  <img alt="p_k = n_k / N" src="https://latex.codecogs.com/png.latex?\large%20p_k%20=%20\frac{n_k}{N}" />
</p>

**Gini index**
<p align="center">
  <img alt="Gini" src="https://latex.codecogs.com/png.latex?\large%20G(p)%20=%201%20-%20\sum_{k=1}^C%20p_k^2" />
</p>

**Entropy**
<p align="center">
  <img alt="Entropy" src="https://latex.codecogs.com/png.latex?\large%20H(p)%20=%20-%20\sum_{k=1}^C%20p_k%20\log_2%20p_k" />
</p>

- Minimum is **0** (pure node).  
- Entropy maximum is **log₂ C** (all classes equally likely).

---

## 2) Information Gain

**Parent impurity**
<p align="center">
  <img alt="Parent impurity" src="https://latex.codecogs.com/png.latex?\large%20I(P)%20=%20\mathrm{crit}(p_P)" />
</p>

**Weighted children impurity**
<p align="center">
  <img alt="Children impurity" src="https://latex.codecogs.com/png.latex?\large%20I_{\text{children}}%20=%20\frac{N_L}{N}\,I(L)%20+%20\frac{N_R}{N}\,I(R)" />
</p>

**Information gain**
<p align="center">
  <img alt="Gain" src="https://latex.codecogs.com/png.latex?\large%20\mathrm{Gain}%20=%20I(P)%20-%20I_{\text{children}}" />
</p>

We pick the split (feature + threshold) that **maximizes** Gain.

---

## 3) Split Thresholds

For each feature `x_d`:  
- Sort unique values `{v_1, ... ,v_m}`.  
- Use midpoints as candidate thresholds.

<p align="center">
  <img alt="Threshold midpoints" src="https://latex.codecogs.com/png.latex?\large%20t_j%20=%20\frac{v_j%20+%20v_{j+1}}{2}" />
</p>

---

## 4) Stopping Conditions

A node becomes a **leaf** if any is true:
- Depth limit reached (e.g., `depth ≥ max_depth`)
- Too few samples to split (`N < min_samples_split`)
- Node is pure (all labels identical)

---

## 5) Leaf Prediction

**Majority class / class-probability prediction**
<p align="center">
  <img alt="Leaf prediction" src="https://latex.codecogs.com/png.latex?\large%20\hat{y}%20=%20\arg\max_k%20p_k" />
</p>

---

## 6) Decision Function (Traversal)

Each internal node compares **one feature** to a threshold:
- If `x_d <= t` → go **left**
- Else → go **right**  
Repeat until a **leaf** is reached; return its class/probabilities.

---

## 7) Complexity

**Training (rough)**  
<p align="center">
  <img alt="Training complexity" src="https://latex.codecogs.com/png.latex?\large%20\mathcal{O}(N\,D\,\log%20N)" />
</p>

**Prediction (tree depth \(h\))**  
<p align="center">
  <img alt="Prediction complexity" src="https://latex.codecogs.com/png.latex?\large%20\mathcal{O}(h)" />
</p>

---

### ✅ Summary
- Trees split space using **axis-aligned thresholds** to reduce impurity.  
- Split choice is based on **impurity reduction (information gain)**.  
- Leaves store **class probabilities** and **majority class**.  
- Boundaries look **boxy** (rectangular regions), unlike linear models.
