# Linear Regression — Math Derivation

i. **Model**  
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{y}_i=w^Tx_i+b" />
</p>

ii. **Loss (sum of squared errors over dataset)**  
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L=\sum_{i=1}^N(\hat{y}_i-y_i)^2" />
</p>

iii. **Closed-form solution (Normal Equation)**  
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?w^*=(X^TX)^{-1}X^Ty" />
</p>

iv. **Gradient of loss wrt parameters**  

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\frac{\partial%20L}{\partial%20w}=\sum_{i=1}^N(\hat{y}_i-y_i)x_i" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\frac{\partial%20L}{\partial%20b}=\sum_{i=1}^N(\hat{y}_i-y_i)" />
</p>

v. **Matrix form of gradient**  
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_wL=X^T(Xw-y)" />
</p>
