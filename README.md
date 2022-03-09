# LDWPSO CNN

Python implementation of LDWPSO CNN (Linearly Decreasing Particle Swarm Optimization Convolutional Neural Network).  
The program rewrites and uses part of the Hyperactive library.

Article: [Optimization of Convolutional Neural Network Using the Linearly Decreasing Weight Particle Swarm Optimization](https://arxiv.org/abs/2001.05670)

# Usage

## Requirement

* Python 3.8
* scikit leran 1.0
* keras 2.8

## LDWPSO CNN

1. Clone this repository

2. The baseline LeNet-5 can be run with the following command

    ```
    python src/lenet5.py
    ```

3. If you use MNIST dataset, you can use the following command.

    ```
    python src/pso_mnist.py
    ```

4. If you use CIFAR-10 dataset, you can use the following command.

    ```
    python src/pso_cifar10.py
    ```

## Validation

The output parameters can be verified by rewriting the following program.

- src/pso_mnist_result.py
- src/pso_cifar10_result.py

## Use data

- MNIST
- CIFAR-10
