//
// ModelInference.cs
//
// Handles model inference using Variational Message Passing
//

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Results from model inference
    /// </summary>
    public class InferenceResults
    {
        public double[][] Dictionary { get; set; } = Array.Empty<double[]>();
        public double[][] CoefficientsMeans { get; set; } = Array.Empty<double[]>();
        public Gamma NoisePrecisionPosterior { get; set; }
    }

    /// <summary>
    /// Handles Bayesian inference for the dictionary learning model
    /// </summary>
    public class ModelInference
    {
        private readonly ModelDefinition _model;
        private readonly InferenceEngine _engine;
        private IGeneratedAlgorithm? _compiledAlgorithm;

        /// <summary>
        /// Creates a new inference engine for the model
        /// </summary>
        public ModelInference(ModelDefinition model)
        {
            _model = model;
            _engine = new InferenceEngine
            {
                Algorithm = new VariationalMessagePassing(),
                ShowProgress = true
            };

            // Optimize for dictionary atoms, coefficients, and noise precision
            _engine.OptimiseForVariables = new IVariable[]
            {
                _model.Dictionary,
                _model.Coefficients,
                _model.NoisePrecision
            };
        }

        /// <summary>
        /// Initializes model variables with random values or custom initialization
        /// </summary>
        /// <param name="numSignals">Number of signals</param>
        /// <param name="numBases">Number of dictionary bases</param>
        /// <param name="signalWidth">Width of each signal</param>
        /// <param name="seed">Random seed for random initialization</param>
        /// <param name="customDictionary">Optional custom dictionary initialization (numBases × signalWidth)</param>
        /// <param name="customCoefficients">Optional custom coefficients initialization (numSignals × numBases)</param>
        public void InitializeVariables(
            int numSignals, 
            int numBases, 
            int signalWidth, 
            int seed = 42,
            double[][]? customDictionary = null,
            double[][]? customCoefficients = null)
        {
            var random = new Random(seed);

            // Initialize dictionary atoms
            var dictionaryInit = new Gaussian[numBases, signalWidth];
            if (customDictionary != null)
            {
                // Use custom initialization
                for (int k = 0; k < numBases; k++)
                    for (int j = 0; j < signalWidth; j++)
                        dictionaryInit[k, j] = Gaussian.FromMeanAndPrecision(customDictionary[k][j], 10.0);
                Console.WriteLine($"Initialized dictionary from custom file");
            }
            else
            {
                // Random initialization
                for (int k = 0; k < numBases; k++)
                    for (int j = 0; j < signalWidth; j++)
                    {
                        double initMean = (random.NextDouble() - 0.5) * 0.2; // Range: [-0.1, 0.1]
                        dictionaryInit[k, j] = Gaussian.FromMeanAndPrecision(initMean, 10.0);
                    }
                Console.WriteLine($"Initialized dictionary with random values in range [-0.1, 0.1]");
            }

            // Initialize coefficients
            var coefficientsInit = new Gaussian[numSignals, numBases];
            if (customCoefficients != null)
            {
                // Use custom initialization
                for (int i = 0; i < numSignals; i++)
                {
                    for (int k = 0; k < numBases; k++)
                    {
                        coefficientsInit[i, k] = Gaussian.FromMeanAndPrecision(customCoefficients[i][k], 10.0);
                    }
                }
                Console.WriteLine($"Initialized coefficients from custom file");
            }
            else
            {
                // Random initialization
                for (int i = 0; i < numSignals; i++)
                {
                    for (int k = 0; k < numBases; k++)
                    {
                        double initMean = (random.NextDouble() - 0.5) * 0.2; // Range: [-0.1, 0.1]
                        coefficientsInit[i, k] = Gaussian.FromMeanAndPrecision(initMean, 10.0);
                    }
                }
                Console.WriteLine($"Initialized coefficients with random values in range [-0.1, 0.1]");
            }

            // Set initial distributions for the variables
            _model.Dictionary.InitialiseTo(Distribution<double>.Array(dictionaryInit));
            _model.Coefficients.InitialiseTo(Distribution<double>.Array(coefficientsInit));
        }

        /// <summary>
        /// Compiles the inference algorithm
        /// </summary>
        public void Compile()
        {
            _compiledAlgorithm = _engine.GetCompiledInferenceAlgorithm(
                new IVariable[] { _model.Dictionary, _model.Coefficients, _model.NoisePrecision });
            Console.WriteLine("Model compiled successfully");
        }

        /// <summary>
        /// Runs inference for up to maxIterations, stopping early when the max absolute
        /// change in Dictionary and Coefficients posterior means drops below tolerance.
        /// </summary>
        public void RunInference(int maxIterations = 100, double tolerance = 1e-3)
        {
            if (_compiledAlgorithm == null)
                throw new InvalidOperationException("Model must be compiled before running inference");

            Console.WriteLine("Running Variational Message Passing...");
            _compiledAlgorithm.Reset();

            double[]? prevDictFlat = null;
            double[]? prevCoeffFlat = null;

            for (int iteration = 1; iteration <= maxIterations; iteration++)
            {
                _compiledAlgorithm.Update(1);

                // Extract current posterior means for convergence check
                var dictPost = _compiledAlgorithm.Marginal<Gaussian[,]>(_model.Dictionary.NameInGeneratedCode);
                var coeffPost = _compiledAlgorithm.Marginal<Gaussian[,]>(_model.Coefficients.NameInGeneratedCode);

                var currDictFlat = FlattenMeans(dictPost);
                var currCoeffFlat = FlattenMeans(coeffPost);

                if (prevDictFlat != null)
                {
                    double maxRelChange = Math.Max(
                        MaxRelChange(currDictFlat, prevDictFlat),
                        MaxRelChange(currCoeffFlat, prevCoeffFlat!));
                    if (iteration % 10 == 0 || iteration == 1)
                        Console.WriteLine($"  Iteration {iteration}, max rel change: {maxRelChange:E3}");

                    if (maxRelChange < tolerance)
                    {
                        Console.WriteLine($"  Converged at iteration {iteration} (max rel change {maxRelChange:E3} < tolerance {tolerance:E3})");
                        return;
                    }
                }
                else if (iteration % 10 == 0 || iteration == 1)
                {
                    Console.WriteLine($"  Iteration {iteration}");
                }

                prevDictFlat = currDictFlat;
                prevCoeffFlat = currCoeffFlat;
            }

            Console.WriteLine($"  Completed {maxIterations} iterations without convergence");
        }

        private static double[] FlattenMeans(Gaussian[,] posteriors)
        {
            int rows = posteriors.GetLength(0);
            int cols = posteriors.GetLength(1);
            var flat = new double[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    flat[i * cols + j] = posteriors[i, j].GetMean();
            return flat;
        }

        /// <summary>
        /// Max absolute change normalised by the RMS of the current values,
        /// making the criterion scale-independent across arrays with different magnitudes.
        /// </summary>
        private static double MaxRelChange(double[] curr, double[] prev)
        {
            double sumSq = 0.0;
            for (int i = 0; i < curr.Length; i++) sumSq += curr[i] * curr[i];
            double scale = Math.Sqrt(sumSq / curr.Length) + 1e-10;

            double max = 0.0;
            for (int i = 0; i < curr.Length; i++)
            {
                double d = Math.Abs(curr[i] - prev[i]) / scale;
                if (d > max) max = d;
            }
            return max;
        }

        /// <summary>
        /// Extracts posterior distributions from the inference results
        /// </summary>
        public InferenceResults GetResults()
        {
            if (_compiledAlgorithm == null)
                throw new InvalidOperationException("Inference must be run before extracting results");

            Console.WriteLine("Extracting posterior distributions...");

            // Get posterior distributions
            var dictionaryPosterior2D = _compiledAlgorithm.Marginal<Gaussian[,]>(_model.Dictionary.NameInGeneratedCode);
            var coefficientsPosterior = _compiledAlgorithm.Marginal<Gaussian[,]>(_model.Coefficients.NameInGeneratedCode);
            var noisePrecisionPosterior = _compiledAlgorithm.Marginal<Gamma>(_model.NoisePrecision.NameInGeneratedCode);

            // Convert to jagged arrays and extract means
            var dictionaryPosterior = ArrayHelpers.GetMeans(ArrayHelpers.ToJagged(dictionaryPosterior2D)!);
            var coefficientsMeansPosterior = ArrayHelpers.GetMeans(ArrayHelpers.ToJagged(coefficientsPosterior)!);

            Console.WriteLine($"Dictionary posterior: {dictionaryPosterior2D.GetLength(0)} × {dictionaryPosterior2D.GetLength(1)}");
            Console.WriteLine($"Coefficients posterior: {coefficientsPosterior.GetLength(0)} × {coefficientsPosterior.GetLength(1)}");

            return new InferenceResults
            {
                Dictionary = dictionaryPosterior,
                CoefficientsMeans = coefficientsMeansPosterior,
                NoisePrecisionPosterior = noisePrecisionPosterior
            };
        }
    }
}
