# SDMLP: Sparse Distributed Memory Multi-Layer Perceptron for Continual Learning

## Description
This repository contains the implementation of the **SDMLP (Sparse Distributed Memory Multi-Layer Perceptron)** architecture, inspired by the paper **"Sparse Distributed Memory is a Continual Learner"**. This model is designed for continual learning tasks and incorporates several novel features:
- **Top-K Activation**: Activates only the top-K neurons during learning, preventing catastrophic forgetting.
- **GABA Switch Mechanism**: Excites all neurons early in training to prevent dead neurons, allowing all neurons to participate initially.
- **Momentum-Free Optimization**: Ensures that inactive neurons are not wrongly updated by eliminating the use of momentum in optimization.

The model has been tested on CIFAR-10, CIFAR-100, and MNIST datasets in a class-incremental learning setup.

---
