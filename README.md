# SDMLP: Sparse Distributed Memory Multi-Layer Perceptron for Continual Learning

## Description
This repository contains the implementation of the **SDMLP (Sparse Distributed Memory Multi-Layer Perceptron)** architecture, inspired by the paper **"Sparse Distributed Memory is a Continual Learner"**. This model is designed for continual learning tasks and incorporates several novel features:
- **Top-K Activation**: Activates only the top-K neurons during learning, preventing catastrophic forgetting.
- **GABA Switch Mechanism**: Excites all neurons early in training to prevent dead neurons, allowing all neurons to participate initially.
- **Momentum-Free Optimization**: Ensures that inactive neurons are not wrongly updated by eliminating the use of momentum in optimization.
-K Value Frontier
The model has been tested on CIFAR-10 datasets in a class-incremental learning setup.

---
# Training vs Continual Learning Tradeoff Analysis- KValueFrontier.ipynb

## Overview
These graphs demonstrate a fundamental tradeoff in machine learning systems between initial training performance and continual learning capabilities. The analysis uses different k-values to explore how model architectures balance these competing objectives.

## Key Findings
The first graph shows a clear inverse relationship between **ImageNet32 training accuracy** and **Split CIFAR10 continual learning performance** across various k-values (ranging from 1 to 250). As k-values increase, continual learning performance deteriorates while initial training accuracy shows modest improvements. This indicates that architectural choices optimizing for traditional training metrics may impair a model's ability to learn new tasks over time.

(![Training vs Continual Learning Performance](https://github.com/user-attachments/assets/b4cb74c2-f44a-4414-b4f2-d75f30e8aca2))

## Pareto Analysis
The second graph presents a **Pareto frontier** of achievable performance combinations between training and continual learning metrics. The frontier's shape confirms that it's impossible to simultaneously maximize both objectives - improvements in one metric necessarily come at the cost of the other. This relationship appears to be fundamental rather than an artifact of current training methods.

![Pareto Frontier of Training and Continual Learning](path/to/your/second-graph.png)

## Implications
These findings have significant implications for designing ML systems that need to maintain plasticity while preserving performance on original tasks. We must carefully choose k-values based on their specific requirements:
- **Lower k-values**: Favor continual learning capability
- **Higher k-values**: Prioritize initial training performance

This tradeoff should inform architectural decisions in scenarios where both initial performance and adaptability are important considerations.
