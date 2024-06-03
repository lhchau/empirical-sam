# Personal Project
Explore Sharpness-Aware Minimization optimizer.

## Research Question: Could we attain similar effect as SAM with only single gradient computation?
Decompose a gradient vector into 2 components:
- Direction
- Magnitude

### Experiment 1: Maintain SAM magnitude, replace these direction with SGD direction
- The results approximate SAM.

### Experiment 2: Maintain SAM direction, replace these magnitude with SGD magnitude
- The results are completely different with SAM, loss landscapes are extremely sharp
