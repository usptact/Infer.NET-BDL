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
  - Sparse mode: `a = 0.5`, `b = 3e-6` (encourages ~70% sparsity, matching typical sparse coefficients)
  - Non-sparse mode: `a = 1.0`, `b = 1.0` (standard prior)
  - This balances sparsity induction with good reconstruction quality
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
- **D**: Dictionary matrix (`numBases × signalWidth`) - learned (dense)
- **C**: Coefficient matrix (`numSignals × numBases`) - learned (sparse)
- **ε**: Gaussian noise with precision `β`

The sparse prior on coefficients (when `a=0.5, b=3e-6`) implements Automatic Relevance Determination (ARD), which automatically drives unnecessary coefficients toward zero while maintaining approximately 70% sparsity.

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
- **Sparse mode**: `a = 0.5`, `b = 3e-6` (encourages ~70% sparsity)
- **Non-sparse mode**: `a = 1.0`, `b = 1.0` (standard prior)
- **Dictionary precision**: `Gamma(1, 1)`
- **Noise precision**: `Gamma(1, 1)`

## Inference

The implementation uses **Variational Message Passing (VMP)** from Infer.NET.

The VMP algorithm approximates the true posterior with a factorized distribution, making inference tractable for large-scale problems.

## Usage

The program supports both synthetic data generation and loading real data from CSV files via command-line arguments.

### Quick Start

Generate and analyze synthetic data with default parameters:
```bash
dotnet run -- --bases 8
```

### Command-Line Options

```
-d, --data <file>        Path to CSV file (one signal per row). If not specified,
                         synthetic data will be generated.
-s, --signals <n>        Number of signals for synthetic data (default: 200)
-b, --bases <n>          Number of dictionary atoms to learn (required)
-w, --width <n>          Signal dimension (auto-detected from CSV)
--sparse <bool>          Use sparse priors (default: true)
-a, --prior-shape <n>    Gamma prior shape parameter (default: 0.5)
--prior-rate <n>         Gamma prior rate parameter (default: 3e-6)
-i, --iterations <n>     Max VMP iterations (default: 100)
--seed <n>               Random seed (default: 42)
--noise <n>              Noise std dev for synthetic data (default: 0.1)
-o, --output-prefix <s>  Prefix for output files (default: "")
-v, --verbose            Enable verbose output
--init-dictionary <file> Path to CSV file with initial dictionary values (numBases × signalWidth)
--init-coefficients <file> Path to CSV file with initial coefficient values (numSignals × numBases)
--help                   Display help
```

### Example Usage

**1. Synthetic data with default sparse prior:**
```bash
dotnet run -- --bases 8 --signals 200 --iterations 100
```

**2. Load data from CSV file:**
```bash
dotnet run -- --data mydata.csv --bases 10 --iterations 50
```

**3. Custom sparse prior (less aggressive):**
```bash
dotnet run -- --bases 8 --prior-shape 1.0 --prior-rate 0.01
```

**4. Multiple experiments with output prefixes:**
```bash
dotnet run -- --bases 8 --iterations 100 --output-prefix "exp1_"
dotnet run -- --bases 12 --iterations 100 --output-prefix "exp2_"
```

**5. Non-sparse mode:**
```bash
dotnet run -- --bases 8 --sparse false
```

**6. Custom initialization (warm start):**
```bash
# Use previous results as initialization
dotnet run -- --bases 8 --init-dictionary prev_learned_dictionary.csv --init-coefficients prev_learned_coefficients.csv
```

**7. Partial custom initialization:**
```bash
# Initialize only dictionary, coefficients will be random
dotnet run -- --bases 8 --init-dictionary my_dictionary.csv
```

### CSV Data Format

**Input signal data** should contain one signal per row:
```csv
0.123,0.456,0.789,...
0.234,0.567,0.890,...
...
```

**Initialization matrices** follow the same format:
- **Coefficients initialization**: `numSignals` rows × `numBases` columns
- **Dictionary initialization**: `numBases` rows × `signalWidth` columns

Requirements:
- Comma-separated values
- All rows must have the same length
- No headers
- Dimensions must match expected model parameters

### Custom Initialization Use Cases

Custom initialization is useful for:

1. **Warm Start**: Resume inference from previous results for faster convergence
```bash
# Run initial inference
dotnet run -- --bases 8 --iterations 50 --output-prefix "init_"
# Resume with more iterations
dotnet run -- --bases 8 --iterations 200 --init-dictionary init_learned_dictionary.csv --init-coefficients init_learned_coefficients.csv
```

2. **Transfer Learning**: Use dictionary learned from one dataset on another
```bash
# Learn dictionary from dataset A
dotnet run -- --data datasetA.csv --bases 10 --output-prefix "dictA_"
# Apply to dataset B with pretrained dictionary
dotnet run -- --data datasetB.csv --bases 10 --init-dictionary dictA_learned_dictionary.csv
```

3. **Informed Initialization**: Use domain knowledge to initialize with known patterns
```bash
# Initialize with Fourier basis or other domain-specific atoms
dotnet run -- --bases 8 --init-dictionary fourier_basis.csv
```

4. **Comparing Initialization Strategies**: Test different starting points
```bash
# Random initialization
dotnet run -- --bases 8 --seed 42 --output-prefix "random_"
# PCA-based initialization
dotnet run -- --bases 8 --init-dictionary pca_basis.csv --output-prefix "pca_"
```

### Output Files

The program saves results to CSV files:
- `[prefix]learned_dictionary.csv`: Learned dictionary atoms (numBases × signalWidth)
- `[prefix]learned_coefficients.csv`: Learned sparse coefficients (numSignals × numBases)
- `[prefix]reconstructed_signals.csv`: Reconstructed signals from learned dictionary
- `[prefix]true_dictionary.csv`: Ground truth (synthetic data only)
- `[prefix]true_coefficients.csv`: Ground truth (synthetic data only)

## Requirements

- .NET 8.0 SDK
- Microsoft.ML.Probabilistic (v0.4.2504.701)
- Microsoft.ML.Probabilistic.Compiler (v0.4.2504.701)
- CommandLineParser (v2.9.1)

## Project Structure

```
Infer.NET-BDL/
├── Program.cs                   # Main entry point with CLI orchestration
├── CommandLineOptions.cs        # Command-line argument definitions
├── ModelDefinition.cs           # Bayesian model structure
├── ModelInference.cs            # VMP inference engine
├── SyntheticDataGenerator.cs    # Synthetic data generation
├── DataLoader.cs                # CSV data loading
├── ErrorMetrics.cs              # Reconstruction error computation
├── ArrayHelpers.cs              # Array utility functions
├── FileManager.cs               # File I/O operations
├── InferNetBDL.csproj           # Project file with dependencies
├── LICENSE                      # Apache 2.0 license
└── README.md                    # This file
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
