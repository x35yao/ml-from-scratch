# Logistic Regression — Math Derivation

i. **Model**  
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?p_i%20=%20\sigma(z_i),%20%5Cquad%20z_i%20=%20w%5ETx_i%20+%20b,%20%5Cquad%20%5Csigma(z)%20=%20%5Cfrac{1}{1+e%5E{-z}}" />
  </p>

  ii. **Likelihood of dataset**  
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cmathcal{L}(w,b)%20=%20%5Cprod_%7Bi=1%7D%5EN%20p_i%5E%7By_i%7D%20%5Ccdot%20(1-p_i)%5E%7B(1-y_i)%7D" />
  </p>

  iii. **Loss over dataset (sum)**  
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?L%20=%20-%20%5Csum_%7Bi=1%7D%5EN%20%5B%20y_i%20%5Clog(p_i)%20+%20(1-y_i)%20%5Clog(1-p_i)%20%5D" />
  </p>

  iv. **Derivative wrt prediction**  
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20p_i%7D%20=%20-%5Cfrac%7By_i%7D%7Bp_i%7D%20+%20%5Cfrac%7B1-y_i%7D%7B1-p_i%7D" />
  </p>

  v. **Sigmoid derivative**  
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7Bdp_i%7D%7Bdz_i%7D%20=%20p_i(1-p_i)" />
  </p>

  vi. **Chain rule (loss wrt logit)**  
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z_i%7D%20=%20%5Cleft(-%5Cfrac%7By_i%7D%7Bp_i%7D%20+%20%5Cfrac%7B1-y_i%7D%7B1-p_i%7D%5Cright)%20%5Ccdot%20p_i(1-p_i)%20=%20p_i-y_i" />
  </p>

  vii. **Gradients wrt parameters**  
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w%7D%20=%20%5Csum_%7Bi=1%7D%5EN%20(p_i-y_i)x_i" />
  </p>
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%7D%20=%20%5Csum_%7Bi=1%7D%5EN%20(p_i-y_i)" />
  </p>
