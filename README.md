# Linear Regression from Scratch using NumPy

This project implements **Multivariate Linear Regression** from scratch using **NumPy**, without relying on machine learning libraries such as Scikit-learn. The goal is to understand the mathematical foundations of linear regression, including cost computation, gradient calculation, and parameter optimization using gradient descent.

## Features

* Implementation of the Linear Regression hypothesis function
* Mean Squared Error (MSE) cost function
* Vectorized gradient computation for multiple features
* Batch Gradient Descent optimization
* Cost history tracking to monitor convergence
* Fully implemented using NumPy for efficient matrix operations

## Mathematical Model

Given a dataset with (m) training examples and (n) features:

[
y_hat = XW + b
]

where:

* X = Feature matrix of shape ((m, n))
* W = Weight vector of shape ((n,))
* b = Bias term
* y_hat = Predicted values

### Cost Function

The model minimizes the Mean Squared Error cost

### Gradient Computation


### Gradient Descent Update Rule


## Project Structure

* `compute_cost()` – Calculates the Mean Squared Error cost.
* `compute_gradient()` – Computes gradients with respect to weights and bias.
* `compute_gradient_descent()` – Optimizes model parameters using batch gradient descent.

## Learning Outcomes

This project demonstrates:

* Matrix-based implementation of machine learning algorithms
* Vectorized computations using NumPy
* Gradient-based optimization
* The relationship between calculus and machine learning
* Building a predictive model without high-level ML frameworks

## Future Improvements

* Feature normalization/standardization
* Learning rate scheduling
* Mini-batch and stochastic gradient descent
* Polynomial regression
* Regularization (L1/L2)
* Performance comparison with Scikit-learn's Linear Regression implementation
