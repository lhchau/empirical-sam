# Empirical Study on Sharpness-Aware Minimization (SAM)

## Research Question: 
**Between the magnitude and direction of the gradient, which is more important to SAM?**

We decompose a gradient vector into two components:
- **Direction:** $\frac{\nabla L(w_t)}{|\nabla L(w_t)|}$
- **Magnitude:** $|\nabla L(w_t)|$

Our investigation focuses on which of these components has a more significant impact on the success of SAM.

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

- **Experiment 1: SAMMAGNITUDE**
  - Maintains the SAM magnitude, replacing the direction with SGD direction.
  - The results approximate SAM performance.

- **Experiment 2: SAMDIRECTION**
  - Maintains the SAM direction, replacing the magnitude with SGD magnitude.
  - The results differ significantly from SAM, with extremely sharp minima.

#### Flatness Results (ResNet-18)

| Flatness (ResNet-18) | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.4$ |
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
&= \eta (I + \rho \nabla_{B}^2 L(w_t)) (\nabla_{D} L(w_t) + \epsilon)
\end{align*}
$$

The gradient magnitude is rescaled with weight $I + \rho \nabla_{B}^2 L(w_t)$.

## Conclusion
- This project demonstrates that the gradient magnitude of SAM primarily contributes to its ability to find flat minima.

### Further Questions
- How can we mimic this preconditioning effect with only a single gradient computation?
- How does this preconditioning help in finding flat minima?
- How can we enhance the SAM trajectory based on these observations?

## Cite this repository
If you use this insight in your research, please cite our work:
```bibtex
@misc{empirical2024sam,
  author = {Hoang-Chau Luong},
  title = {Empirical Study on Sharpness-Aware Minimization (SAM)},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lhchau/empirical-sam}},
}
```