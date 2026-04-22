# Neural Network – Training from Scratch

This project implements a Multilayer Perceptron (MLP) neural network with the SGD learning algorithm from scratch in Python. Built entirely without deep learning frameworks (like PyTorch or TensorFlow), this project was developed for educational and experimental purposes to deeply understand the underlying mechanics of neural networks. 

The performance of the simulator was rigorously evaluated using the MONK datasets (1, 2, and 3) as well as the CUP dataset provided by Prof. Micheli. 

A comprehensive analysis of the project, authored by Andres Lazzari and Leonardo Elmi from the Università di Pisa, can be found in the attached technical report (`ML_Elmi_Lazzari.pdf`).

---

## 🚀 Key Features

The project features a modular architecture designed to handle both classification and regression tasks. All operations (forward pass, backward pass, weight updates) are implemented manually.

* **Architecture Flexibility:** Configurable hidden and output layers.
* **Activation Functions:** Support for Sigmoid, ReLU, Leaky ReLU, and Tanh.
* **Weight Initialization:** Utilizes uniform He initialization for ReLU and Leaky ReLU, and uniform Glorot for Sigmoid and Tanh.
* **Learning Strategies:** Capability to seamlessly switch among online, batch, and minibatch learning.
* **Optimization Techniques:**
    * Momentum to add inertia to weight updates.
    * Rudimental learning rate decay proportional to epochs.
    * L2 regularization.
    * Early Stopping with predefined patience thresholds.
* **Validation:** Supports grid search with K-fold cross-validation and stratified hold-out methodologies.

---

## 🛠️ Requirements

Before running the project, ensure you have:

* Python ≥ 3.9
* An active virtual environment (recommended)

Install the necessary libraries via:

```bash
pip install -r requirements.txt
