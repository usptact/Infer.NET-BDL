# Infer.NET-BDL

Bayesian Dictionary Learning using Infer.NET - A probabilistic approach to learning sparse representations of signals.

## Overview

This project implements Bayesian Dictionary Learning (BDL), a probabilistic framework for learning sparse signal representations. Dictionary learning aims to decompose observed signals into a linear combination of dictionary atoms (basis functions) with sparse coefficients. The Bayesian approach provides uncertainty quantification and principled handling of noise and model complexity.

## Motivation

Dictionary learning is a fundamental problem in signal processing, machine learning, and data analysis. Traditional methods (e.g., K-SVD, MOD) use optimization-based approaches that lack uncertainty quantification. Bayesian Dictionary Learning addresses these limitations by:

- **Uncertainty Quantification**: Provides posterior distributions over dictionary elements and coefficients, not just point estimates
- **Automatic Sparsity**: Uses hierarchical priors to automatically encourage sparse representations
- **Noise Modeling**: Explicitly models observation noise with learnable precision
- **Principled Regularization**: Uses Bayesian priors instead of ad-hoc regularization terms

## Model Structure

The model follows a hierarchical Bayesian structure with three levels:

### Level 1: Hyperparameters
- **Coefficient precision hyperparameters**: `a` and `b` (shape and rate for Gamma prior)
  - Sparse mode: `a = 1.0`, `b = 1.0` (provides mild regularization without crushing coefficients)
  - Non-sparse mode: `a = 1.0`, `b = 1.0` (standard prior)
  - Note: Previous setting `a = 0.5`, `b = 1e-6` was too strong and crushed coefficients to near-zero
- **Dictionary precision prior**: `Gamma(1, 1)`
- **Noise precision prior**: `Gamma(1, 1)`

### Level 2: Precisions (Inverse Variances)
- **Coefficient precisions**: `τ_c[signal, basis] ~ Gamma(a, b)`
- **Dictionary precisions**: `τ_d[basis, sample] ~ Gamma(1, 1)`
- **Noise precision**: `β ~ Gamma(1, 1)`

### Level 3: Variables
- **Coefficients**: `c[signal, basis] ~ Gaussian(0, τ_c[signal, basis])`
- **Dictionary**: `d[basis, sample] ~ Gaussian(μ_d[basis, sample], τ_d[basis, sample])`
  - Where `μ_d[basis, sample] ~ Gaussian(0, 0.01)`
- **Observed signals**: `y[signal, sample] ~ Gaussian((D×C)[signal, sample], β)`

### Mathematical Formulation

The model learns the decomposition:

```
Y ≈ D × C + ε
```

Where:
- **Y**: Observed signals matrix (`numSignals × signalWidth`)
- **D**: Dictionary matrix (`numBases × signalWidth`) - learned
- **C**: Coefficient matrix (`numSignals × numBases`) - learned (sparse)
- **ε**: Gaussian noise with precision `β`

The sparse prior on coefficients (when `a=0.5, b=1e-6`) implements Automatic Relevance Determination (ARD), which automatically drives unnecessary coefficients toward zero.

## Parameters

### Model Parameters
- **`numSignals`**: Number of observed signals (default: 200)
- **`numBases`**: Number of dictionary atoms/basis functions (default: 8)
- **`signalWidth`**: Dimension of each signal (default: 64)
- **`sparse`**: Whether to use sparse priors for coefficients (default: true)

### Inference Parameters
- **`maxIterations`**: Maximum number of Variational Message Passing iterations (default: 100)
- **`tolerance`**: Convergence tolerance (default: 1e-3)

### Hyperparameters
- **Sparse mode**: `a = 1.0`, `b = 1.0` (provides mild regularization)
- **Non-sparse mode**: `a = 1.0`, `b = 1.0` (standard prior)
- **Dictionary precision**: `Gamma(1, 1)`
- **Noise precision**: `Gamma(1, 1)`

## Inference

The implementation uses **Variational Message Passing (VMP)** from Infer.NET, which:

1. **Compiles the model**: Converts the probabilistic model into an efficient inference algorithm
2. **Initializes variables**: Sets up observed values and initializes latent variables
3. **Iterative updates**: Performs VMP iterations to approximate the posterior distributions
4. **Extracts posteriors**: Returns posterior distributions over dictionary, coefficients, and noise precision

### Inference Process

1. **Model Definition**: Define the hierarchical Bayesian model using Infer.NET's probabilistic programming constructs
2. **Compilation**: Compile the model into an optimized inference algorithm
3. **Observation**: Set observed signal values `Y`
4. **Iteration**: Run VMP updates until convergence or maximum iterations
5. **Extraction**: Extract posterior means (and variances) for dictionary and coefficients

The VMP algorithm approximates the true posterior with a factorized distribution, making inference tractable for large-scale problems.

## Use Cases

Bayesian Dictionary Learning has applications in:

### Signal Processing
- **Image denoising**: Learn dictionaries of image patches for noise removal
- **Audio processing**: Decompose audio signals into sparse frequency components
- **Compressed sensing**: Recover sparse signals from incomplete measurements

### Machine Learning
- **Feature learning**: Learn interpretable features from raw data
- **Dimensionality reduction**: Represent high-dimensional data with sparse low-dimensional codes
- **Transfer learning**: Learn dictionaries that generalize across domains

### Data Analysis
- **Time series analysis**: Decompose time series into interpretable components
- **Neuroscience**: Analyze neural signals and identify patterns
- **Computer vision**: Learn visual dictionaries for object recognition

### Advantages Over Traditional Methods
- **Uncertainty quantification**: Know how confident the model is about each dictionary element
- **Automatic sparsity**: No need to tune sparsity regularization parameters
- **Noise robustness**: Explicit noise modeling handles noisy observations
- **Bayesian model selection**: Can compare models with different numbers of bases

## Running the Example

```bash
dotnet run
```

The program will:
1. Generate synthetic data (signals, true dictionary, true coefficients)
2. Run Bayesian inference to learn the dictionary and coefficients
3. Save results to CSV files:
   - `learned_dictionary.csv`: Learned dictionary atoms
   - `learned_coefficients.csv`: Learned sparse coefficients
   - `true_dictionary.csv`: True dictionary (for comparison)
   - `true_coefficients.csv`: True coefficients (for comparison)

## Requirements

- .NET 8.0 SDK
- Microsoft.ML.Probabilistic (v0.4.2504.701)
- Microsoft.ML.Probabilistic.Compiler (v0.4.2504.701)

## Project Structure

```
Infer.NET-BDL/
├── Program.cs              # Main program with model definition and inference
├── InferNetBDL.csproj      # Project file with dependencies
└── README.md               # This file
```

## References

- Infer.NET: Microsoft's probabilistic programming framework
- Dictionary Learning: Sparse representation learning
- Variational Message Passing: Approximate Bayesian inference algorithm
- Automatic Relevance Determination: Bayesian sparsity-inducing priors
