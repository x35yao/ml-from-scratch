# 📘 Softmax Regression (Multinomial Logistic Regression)

---

## 🔹 Model

Given input vector $x \in \mathbb{R}^D$, parameters $W \in \mathbb{R}^{K \times D}$, and bias $b \in \mathbb{R}^K$:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?z%20%3D%20Wx%20%2B%20b" />
</p>

where $z \in \mathbb{R}^K$ are the class logits.

The **softmax function** converts logits into probabilities:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?p(y%3Dk%20%5Cmid%20x)%20%3D%20%5Cfrac%7Be%5E%7Bz_k%7D%7D%7B%5Csum_%7Bj%3D1%7D%5EK%20e%5E%7Bz_j%7D%7D" />
</p>

---

## 🔹 Likelihood

For dataset $\{(x_i, y_i)\}_{i=1}^N$ with labels $y_i \in \{1, \dots, K\}$, the probability of the whole dataset is:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?L(W%2C%20b)%20%3D%20%5Cprod_%7Bi%3D1%7D%5EN%20p(y_i%20%5Cmid%20x_i%3B%20W%2C%20b)" />
</p>

Expanding with one-hot encoding $y_{i,k}$:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?L(W%2C%20b)%20%3D%20%5Cprod_%7Bi%3D1%7D%5EN%20%5Cprod_%7Bk%3D1%7D%5EK%20p(y%3Dk%20%5Cmid%20x_i)%5E%7By_%7Bi%2Ck%7D%7D" />
</p>

---

## 🔹 Loss (Negative Log-Likelihood / Cross-Entropy)

Taking the negative log of the likelihood:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D(W%2C%20b)%20%3D%20-%20%5Csum_%7Bi%3D1%7D%5EN%20%5Csum_%7Bk%3D1%7D%5EK%20y_%7Bi%2Ck%7D%20%5Clog%20p(y%3Dk%20%5Cmid%20x_i)" />
</p>

For optimization, we usually average over samples:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D(W%2C%20b)%20%3D%20-%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Csum_%7Bk%3D1%7D%5EK%20y_%7Bi%2Ck%7D%20%5Clog%20p(y%3Dk%20%5Cmid%20x_i)" />
</p>

---

## 🔹 Gradients

Let $P \in \mathbb{R}^{N \times K}$ be predicted probabilities and $Y$ the one-hot label matrix.

Gradient w.r.t. weights $W$:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cnabla_W%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20(P-Y)%5ET%20X" />
</p>

Gradient w.r.t. bias $b$:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cnabla_b%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20(P_i%20-%20Y_i)" />
</p>

---

## 🔹 Algorithm (Gradient Descent)

1. Initialize $W, b$ (zeros or small random values).  
2. Repeat until convergence:
   - Compute logits: $ z = XW^T + b $  
   - Apply softmax: $ P = \text{softmax}(z) $  
   - Compute negative log-likelihood loss  
   - Compute gradients $\nabla_W, \nabla_b$  
   - Update parameters:  
     <p align="center">
     <img src="https://latex.codecogs.com/png.latex?W%20%5Cleftarrow%20W%20-%20%5Calpha%20%5Cnabla_W" />
     </p>  
     <p align="center">
     <img src="https://latex.codecogs.com/png.latex?b%20%5Cleftarrow%20b%20-%20%5Calpha%20%5Cnabla_b" />
     </p>

---

## 🔹 Prediction

For a new input $x$:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Chat%7By%7D%20%3D%20%5Carg%5Cmax_k%20p(y%3Dk%20%5Cmid%20x)" />
</p>
