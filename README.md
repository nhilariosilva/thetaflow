# Thetaflow

Bridging the gap between Statistical Inference and Deep Learning.

Thetaflow is a Python package built on top of TensorFlow/Keras designed to fully integrate statistical modeling with neural network components. It allows researchers and data scientists to define any statistical model where parameters can be:

Dynamic: Modeled as outputs of a complex neural network.
Static: Treated as independent, learnable weights (standard statistical coefficients).

It generalizes Maximum Likelihood Estimation (MLE) for a massive class of models, acting as a flexible optimizer that brings the power of backpropagation to rigorous statistical inference.

Key Features

- Flexible Parameter Definition: seamless mixing of deep learning outputs and scalar statistical parameters.
- Custom Likelihoods: Define any probability density function (PDF) or mass function (PMF) as your objective.
- TensorFlow/Keras Backend: leverages hardware acceleration (GPU/TPU) and automatic differentiation for complex optimization landscapes.
- General Optimizer: Solves for the Maximum Likelihood Estimate (MLE) across arbitrary model architectures.
