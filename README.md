# COMP5012 - Multi-Objective Optimisation for Underwater Mine Detection

## Overview
Implementation of NSGA-II algorithm for solving the underwater mine detection 
assignment problem as part of COMP5012 Computational Intelligence module.

## Problem Description
- **Dataset**: OR-Library assign100.txt (100×100 cost matrix)
- **Objective 1**: Minimize total mission cost
- **Objective 2**: Maximize total detection probability

## Algorithm
NSGA-II (Non-dominated Sorting Genetic Algorithm II) with:
- PMX Crossover (permutation-safe)
- Swap Mutation
- Binary Tournament Selection
- Crowding Distance for diversity

## Usage
```bash
python nsga2_mine_detection.py
```

## Requirements
- Python 3.x
- NumPy
- Matplotlib

## Author
25126736 
University of Plymouth - COMP5012
