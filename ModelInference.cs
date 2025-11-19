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
        public double[][] DictionaryMeans { get; set; } = Array.Empty<double[]>();
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

            // Optimize for dictionaryMeans (which determines dictionary), coefficients, and noise precision
            _engine.OptimiseForVariables = new IVariable[] 
            { 
                _model.DictionaryMeans, 
                _model.Coefficients, 
                _model.NoisePrecision 
            };
        }

        /// <summary>
        /// Initializes model variables with random values to break symmetry
        /// </summary>
        public void InitializeVariables(int numSignals, int numBases, int signalWidth, int seed = 42)
        {
            var random = new Random(seed);

            // Initialize dictionary means with small random values
            var dictionaryMeansInit = new Gaussian[numBases][];
            for (int k = 0; k < numBases; k++)
            {
                dictionaryMeansInit[k] = new Gaussian[signalWidth];
                for (int j = 0; j < signalWidth; j++)
                {
                    // Small random initialization: mean ~ N(0, 0.1), precision = 10
                    double initMean = (random.NextDouble() - 0.5) * 0.2; // Range: [-0.1, 0.1]
                    dictionaryMeansInit[k][j] = Gaussian.FromMeanAndPrecision(initMean, 10.0);
                }
            }

            // Initialize coefficients with small random values
            var coefficientsInit = new Gaussian[numSignals, numBases];
            for (int i = 0; i < numSignals; i++)
            {
                for (int k = 0; k < numBases; k++)
                {
                    // Small random initialization: mean ~ N(0, 0.1), precision = 10
                    double initMean = (random.NextDouble() - 0.5) * 0.2; // Range: [-0.1, 0.1]
                    coefficientsInit[i, k] = Gaussian.FromMeanAndPrecision(initMean, 10.0);
                }
            }

            // Set initial distributions for the variables
            _model.DictionaryMeans.InitialiseTo(Distribution<double>.Array(dictionaryMeansInit));
            _model.Coefficients.InitialiseTo(Distribution<double>.Array(coefficientsInit));

            Console.WriteLine($"Initialized dictionary means with random values in range [-0.1, 0.1]");
            Console.WriteLine($"Initialized coefficients with random values in range [-0.1, 0.1]");
        }

        /// <summary>
        /// Compiles the inference algorithm
        /// </summary>
        public void Compile()
        {
            _compiledAlgorithm = _engine.GetCompiledInferenceAlgorithm(
                new IVariable[] { _model.DictionaryMeans, _model.Coefficients, _model.NoisePrecision });
            Console.WriteLine("Model compiled successfully");
        }

        /// <summary>
        /// Runs inference for the specified number of iterations
        /// </summary>
        public void RunInference(int maxIterations = 100)
        {
            if (_compiledAlgorithm == null)
                throw new InvalidOperationException("Model must be compiled before running inference");

            Console.WriteLine("Running Variational Message Passing...");
            _compiledAlgorithm.Reset();

            for (int iteration = 1; iteration <= maxIterations; iteration++)
            {
                _compiledAlgorithm.Update(1);

                if (iteration % 10 == 0 || iteration == 1)
                {
                    Console.WriteLine($"  Iteration {iteration}");
                }
            }

            Console.WriteLine($"  Completed {maxIterations} iterations");
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
            var dictionaryMeansPosteriorArray = _compiledAlgorithm.Marginal<Gaussian[][]>(_model.DictionaryMeans.NameInGeneratedCode);
            var coefficientsPosterior = _compiledAlgorithm.Marginal<Gaussian[,]>(_model.Coefficients.NameInGeneratedCode);
            var noisePrecisionPosterior = _compiledAlgorithm.Marginal<Gamma>(_model.NoisePrecision.NameInGeneratedCode);

            // Convert to jagged arrays and extract means
            var dictionaryMeansPosterior = ArrayHelpers.GetMeans(dictionaryMeansPosteriorArray);
            var coefficientsMeansPosterior = ArrayHelpers.GetMeans(ArrayHelpers.ToJagged(coefficientsPosterior)!);

            Console.WriteLine($"Dictionary means posterior: {dictionaryMeansPosteriorArray.Length} × {dictionaryMeansPosteriorArray[0].Length}");
            Console.WriteLine($"Coefficients posterior: {coefficientsPosterior.GetLength(0)} × {coefficientsPosterior.GetLength(1)}");
            Console.WriteLine($"Noise precision: {noisePrecisionPosterior}");

            return new InferenceResults
            {
                DictionaryMeans = dictionaryMeansPosterior,
                CoefficientsMeans = coefficientsMeansPosterior,
                NoisePrecisionPosterior = noisePrecisionPosterior
            };
        }
    }
}
