# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a function encoder research codebase that implements neural function approximation using basis functions and neural ODEs. The system represents functions as linear combinations of learned basis functions and includes experiments on dynamical systems like Van der Pol oscillators, Kepler orbits, and polynomial functions.

## Dependencies

Install required dependencies with:
```bash
pip install -r requirements.txt
```

Core dependencies: torch, matplotlib, tqdm, scikit-fda, jupyter

## Code Architecture

### Core Components

- **`function_encoder/`**: Main library code
  - `function_encoder.py`: Core `FunctionEncoder` and `BasisFunctions` classes for function approximation
  - `coefficients.py`: Multiple coefficient computation methods (least squares, LASSO, RLS, gradient descent)
  - `inner_products.py`: Inner product functions for projections
  - `losses.py`: Loss functions for training
  - `model/`: Neural network architectures
    - `mlp.py`: Multi-layer perceptron implementations
    - `neural_ode.py`: Neural ODE implementation with RK4 integrator
  - `utils/`: Utilities for training, experiment saving, and plotting

### Key Classes

- **`FunctionEncoder`**: Main class that encodes functions as linear combinations of basis functions. Takes basis functions, residual function (optional), coefficient computation method, and inner product function.
- **`BasisFunctions`**: Container for multiple basis function modules, evaluates all basis functions and stacks results.
- **`NeuralODE`**: Neural ordinary differential equation solver with customizable ODE functions and integrators.

### Examples Directory

The `examples/` directory contains numerous experiments:
- **Kepler orbit experiments**: `kepler*.py` files for orbital dynamics
- **Van der Pol oscillator**: `van_der_pol*.py`, `vdp*.py` files
- **Polynomial approximation**: `polynomial*.py` files
- **Method comparisons**: Scripts comparing PCA, pruning, progressive methods
- **Plotting utilities**: Various visualization scripts

## Running Experiments

Most experiments are standalone Python scripts that can be run directly:
```bash
python examples/kepler_train_prune.py
python examples/van_der_pol_prune.py
python examples/polynomial_progressive.py
```

Experiments typically:
1. Generate or load datasets for specific dynamical systems
2. Define basis functions (often NeuralODE-based)
3. Train the function encoder
4. Apply pruning or other compression techniques
5. Save results and generate visualizations

## Development Notes

- The codebase uses PyTorch as the primary framework
- Most neural networks follow the MLP architecture pattern from `function_encoder/model/mlp.py`
- Experiments save results to `examples/results/` subdirectories
- Model checkpoints are saved as `.pth` files
- The system supports multiple coefficient computation methods - choose based on the specific requirements (sparsity, stability, etc.)
- Neural ODE integrations use RK4 by default but integrators are configurable