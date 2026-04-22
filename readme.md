# Neural Network – Training from Scratch

[cite_start]This project implements a Multilayer Perceptron (MLP) neural network with the SGD learning algorithm from scratch in Python[cite: 10]. Built entirely without deep learning frameworks (like PyTorch or TensorFlow), this project was developed for educational and experimental purposes to deeply understand the underlying mechanics of neural networks. 

[cite_start]The performance of the simulator was rigorously evaluated using the MONK datasets (1, 2, and 3) as well as the CUP dataset provided by Prof. Micheli[cite: 11]. 

[cite_start]A comprehensive analysis of the project, authored by Andres Lazzari and Leonardo Elmi from the Università di Pisa[cite: 3, 5, 6], can be found in the attached technical report (`ML_Elmi_Lazzari.pdf`).

---

## 🚀 Key Features

[cite_start]The project features a modular architecture designed to handle both classification and regression tasks[cite: 32]. All operations (forward pass, backward pass, weight updates) are implemented manually.

* [cite_start]**Architecture Flexibility:** Configurable hidden and output layers[cite: 45].
* [cite_start]**Activation Functions:** Support for Sigmoid, ReLU, Leaky ReLU, and Tanh[cite: 45].
* [cite_start]**Weight Initialization:** Utilizes uniform He initialization for ReLU and Leaky ReLU, and uniform Glorot for Sigmoid and Tanh[cite: 48].
* [cite_start]**Learning Strategies:** Capability to seamlessly switch among online, batch, and minibatch learning[cite: 46].
* **Optimization Techniques:**
    * [cite_start]Momentum to add inertia to weight updates[cite: 46].
    * [cite_start]Rudimental learning rate decay proportional to epochs[cite: 46].
    * [cite_start]L2 regularization[cite: 47].
    * [cite_start]Early Stopping with predefined patience thresholds[cite: 47].
* [cite_start]**Validation:** Supports grid search with K-fold cross-validation and stratified hold-out methodologies[cite: 52, 53].

---

## 🛠️ Requirements

Before running the project, ensure you have:

* Python ≥ 3.9
* An active virtual environment (recommended)

Install the necessary libraries via:

```bash
pip install -r requirements.txt
