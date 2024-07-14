# Empirical Study on Sharpness-Aware Minimization (SAM)

## Table of Contents
1. [Research Question](#research-question)
2. [Optimization Background](#optimization-background)
    - [Basic Concepts](#basic-concepts)
    - [Stochasitc Gradient Descent](#stochasitc-gradient-descent)
    - [Challenges in Optimization](#challenges-in-optimization)
    - [Sharpness-Aware Minimization (SAM)](#sharpness-aware-minimization-sam)
3. [Experiments](#experiments)
4. [Observations and Reproducibility](#observations-and-reproducibility)
5. [Intuition from Update Rule](#intuition-from-update-rule)
6. [Conclusion](#conclusion)
    - [Further Questions](#further-questions)
7. [Cite this repository](#cite-this-repository)

## Research Question: 
**Between the magnitude and direction of the gradient, which is more important to SAM?**

We decompose a gradient vector into two components:
- **Direction:** $\frac{\nabla L(w_t)}{|\nabla L(w_t)|}$
- **Magnitude:** $|\nabla L(w_t)|$

Our investigation focuses on which of these components has a more significant impact on the success of SAM.

## Optimization Background

Optimization plays a critical role in training deep learning models. Two techniques we discuss here are Stochastic Gradient Descent (SGD) and Sharpness-Aware Minimization (SAM).

### Basic Concepts

Optimization in the context of machine learning refers to the process of adjusting the parameters of a model to minimize (or maximize) an objective function, often referred to as the loss function. The goal is to find the parameter values that result in the best performance of the model on a given task.

### Stochastic Gradient Descent (SGD)

SGD is a fundamental optimization algorithm used to minimize the loss function. The key idea is to iteratively adjust the model parameters in the direction opposite to the gradient of the loss function with respect to the parameters. This process is repeated until convergence.

The update rule for stochastic gradient descent is given by:
$w_{t+1} = w_t - \eta \nabla L(w_t)$
where:
- $w_t$ are the model parameters at iteration $t$,
- $\eta$ is the learning rate,
- $\nabla L(w_t)$ is the gradient of the loss function with respect to $w_t$.

### Challenges in Optimization

- **Local Minima**: Optimization algorithms can get stuck in local minima, especially in non-convex loss landscapes.
- **Saddle Points**: Points where the gradient is zero but are not minima can slow down optimization.
- **Sharp Minima**: Solutions where the loss function has steep curves can lead to poor generalization on new data.

### Sharpness-Aware Minimization (SAM)

Sharpness-Aware Minimization (SAM) aims to address the issue of sharp minima. Sharp minima are regions where the loss function has steep slopes, which often correspond to poor generalization. SAM seeks to find flatter solutions that are robust to perturbations in the model parameters, leading to better generalization.

SAM modifies the loss function to penalize sharp minima: $L^{SAM}(w_t) = \max_{ || \epsilon || \leq \rho } L(w_t + \epsilon)$

To solve this objective function, SAM proposed the update rule as:
$$
w_{t+1} = w_t - \eta \nabla L(w_t + \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) || })
$$

## Experiments

### Experiment Setting
- Batch size: 256
- Initial learning rate (lr): 0.1
- Momentum: 0.9
- Weight decay (wd): 0.001
- $\rho \in \{0.1, 0.2, 0.4\}$

#### Accuracy Results (ResNet-18)

| Accuracy (ResNet-18) | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.4$ |
|----------------------|--------------|--------------|--------------|
| SAM                  | 79.24        | 79.54        | 79.44        |
| SAMDIRECTION         | 77.89        | 78.52        | 78.22        |
| SAMMAGNITUDE         | 78.73        | 79.23        | 77.94        |

| Accuracy (ResNet-34) | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.4$ |
|----------------------|--------------|--------------|--------------|
| SAM                  | ??        | 80.95        | 80.89        |
| SAMDIRECTION         | 79.35        | 80.02        | 76.38        |
| SAMMAGNITUDE         | 80.08        | 79.71        | 72.74        |

| Accuracy (ResNet-50) | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.4$ |
|----------------------|--------------|--------------|--------------|
| SAM                  | ??        | 81.24        | ??        |
| SAMDIRECTION         | ??        | ??        | 80.34        |
| SAMMAGNITUDE         | ??        | ??        | ??        |

| Accuracy (WideResNet-28-10) | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.4$ |
|----------------------|--------------|--------------|--------------|
| SAM                  | 83.50        | 83.91        | 83.44        |
| SAMDIRECTION         | 82.44        | 82.54        | 82.11        |
| SAMMAGNITUDE         | 82.47        | 80.63        | 79.36        |

- **Experiment 1: SAMMAGNITUDE**
  - Maintains the SAM magnitude, replacing the direction with SGD direction.
  - The results approximate SAM performance.

- **Experiment 2: SAMDIRECTION**
  - Maintains the SAM direction, replacing the magnitude with SGD magnitude.
  - The results differ significantly from SAM, with extremely sharp minima.

#### Flatness Results (ResNet-18)

| Sharpness (ResNet-18) | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.4$ |
|----------------------|--------------|--------------|--------------|
| SAM                  | 71.04        | 56.53        | 52.23        |
| SAMDIRECTION         | 259.27       | 3737.54      | 1254.17      |
| SAMMAGNITUDE         | 93.54        | 114.01       | 373.49       |

- **Experiment 3: Magnitude Comparison**
  - The magnitude of SAM updates is larger than that of SGD updates.
  - We count the number of instances where the ratio of SAM update over SGD update is greater than one.
  - Results indicate that this ratio is over 50% during initial training and increases to 85% at later stages.

## Observations and Reproducibility
- **The gradient magnitude** is a crucial factor in determining the ability of SAM to find flat minima.

To reproduce our experiments, follow these steps:
- Install the `wandb` package:
```
pip install wandb
pip install -e .
```
- Run the scripts:
```
# SAM
python sam_hessian/train.py --experiment=default_sam --rho=0.2 --wd=0.001 --project_name=CIFAR100-SAM --framework_name=wandb

# SAMDIRECTION
python sam_hessian/train.py --experiment=default_sam --opt_name=samdirection --rho=0.2 --wd=0.001 --project_name=CIFAR100-SAM --framework_name=wandb

# SAMMAGNITUDE
python sam_hessian/train.py --experiment=default_sam --opt_name=sammagnitude --rho=0.2 --wd=0.001 --project_name=CIFAR100-SAM --framework_name=wandb
```

## Intuition from Update Rule
Considering the SAM update rule, we denote \( D \) as the gradient computed on the full batch, and \( B \) as the gradient computed on a mini-batch:

$$
w_{t+1} = w_t - \eta \nabla_{B} L(w_t + \rho \frac{\nabla_{B} L(w_t)}{||\nabla_{B} L(w_t)||})
$$

We focus on the gradient and use a first-order Taylor approximation. For convenience in analysis, and without loss of generality, we eliminate the gradient normalization in the denominator:

$$
\begin{align*}
\eta \nabla_{B} L(w_t + \rho \nabla_{B} L(w_t)) &= \eta [\nabla_{B} L(w_t) + \rho \nabla_{B}^2 L(w_t) \nabla_{B} L(w_t)] \\
&= \eta (I + \rho \nabla_{B}^2 L(w_t)) \nabla_{B} L(w_t) \\
\end{align*}
$$

The gradient magnitude is rescaled with weight $I + \rho \nabla_{B}^2 L(w_t)$.

### Quick Check for Intuition

We introduce an SGD variant named SGDHESS, derived from the following intuition:

$$
w_{t+1} = w_t - \eta \nabla L(w_t) (I + \rho \nabla_{B}^2 L(w_t))
$$
 
For efficient computation, we use the Gauss-Newton approximation $G$ for $\nabla_{B}^2 L(w_t)$. This implementation is based on [AdaHessian](https://github.com/amirgholami/adahessian).

The results demonstrate that modifying the magnitude of the gradient $\nabla L(w_t)$ by considering the Hessian $\nabla_{B}^2 L(w_t)$ can lead to finding flatter minima.

| Flatness (ResNet-18) | Accuracy     | Sharpness    |
|----------------------|--------------|--------------|
| SGD                  | 77.36%       | 207.06       |
| SGDHESS ($\rho = 0.05$) | 77.27%   | 192.45       |
| SGDHESS ($\rho = 1$)    | 76.75%   | 151.62       |
| SGDHESS ($\rho = 2$)    | 75.96%   | 138.62       |

However, the test accuracy of SGDHESS is lower than that of SGD. This phenomenon suggests that the actual SAM update does not only reduce sharpness.

## Conclusion
- This project demonstrates that the gradient magnitude of SAM primarily contributes to its ability to find flat minima.

### Further Questions
- How can we mimic this preconditioning effect with only a single gradient computation?
- How does this preconditioning help in finding flat minima?
- How can we enhance the SAM trajectory based on these observations?

## Cite this repository
If you use this insight in your research, please cite our work:
```bibtex
@techreport{Luong_Empirical_Study_on_2024,
author = {Luong, Hoang-Chau},
month = jun,
title = {{Empirical Study on Sharpness-Aware Minimization}},
url = {https://github.com/lhchau/empirical-sam},
year = {2024}}
```