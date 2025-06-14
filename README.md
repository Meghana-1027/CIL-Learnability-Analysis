# Class Incremental Learning with ROW Framework

## Overview

This project implements a high-performance Class Incremental Learning (CIL) system based on the research paper *“Learnability and Algorithm for Continual Learning”*, which introduces the ROW (Regularized OOD Weighting) framework. The system is designed to incrementally learn from a sequence of classification tasks while retaining knowledge of previously seen classes. It uses task-specific prediction heads and a probabilistic inference mechanism to address the problem of catastrophic forgetting.

## Dataset

The implementation uses **Split-CIFAR-10**, where the standard CIFAR-10 dataset is split into 5 sequential tasks, each containing 2 mutually exclusive classes. This benchmark is widely used in continual learning research to evaluate performance on disjoint class splits.

## Methodology

- A **ResNet18** backbone pretrained on ImageNet is used as a shared feature extractor across all tasks.
- Each task has two specialized heads:
  - A **WP (within-task prediction) head** for predicting in-distribution classes.
  - An **OOD (out-of-distribution detection) head** for identifying whether the input is from the current task or previous tasks.
- A **balanced replay buffer** maintains representative exemplars from previous tasks. During training, data from the current task is mixed with replay samples to reduce forgetting.
- The model is trained using cross-entropy loss for both WP and OOD outputs.
- At inference time, predictions are made using the product of WP and OOD probabilities, eliminating the need for task IDs.

## Results

- The system achieves **over 95% accuracy on newly learned tasks**.
- It retains **70%+ accuracy on earlier tasks**, demonstrating effective mitigation of catastrophic forgetting.
- Task-wise accuracy is plotted after each stage to visualize retention and performance trends across all tasks.

## Conclusion

This implementation validates the ROW framework in a practical continual learning setup. It provides a strong foundation for further experimentation with exemplar replay, probabilistic task inference, and scalable lifelong learning strategies.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib

Install dependencies using:

```bash
pip install torch torchvision matplotlib

