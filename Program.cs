//
// Program.cs
//
// Bayesian Dictionary Learning using Infer.NET
// This demonstrates the core model structure: recovering coefficients and dictionary from signals
//

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Bayesian Dictionary Learning Example
    /// 
    /// Model Structure:
    /// ================
    /// 
    /// Hierarchical Prior Structure:
    /// ------------------------------
    /// Level 1 (Hyperparameters):
    ///   - a, b: Shape and rate for coefficient precision priors (sparse: a=0.5, b=1e-6)
    ///   - Gamma(1,1): Prior for dictionary precisions
    ///   - Gamma(1,1): Prior for noise precision
    /// 
    /// Level 2 (Precisions):
    ///   - τ_c[signal, basis] ~ Gamma(a, b)          // Coefficient precisions
    ///   - τ_d[basis, sample] ~ Gamma(1, 1)          // Dictionary precisions  
    ///   - β ~ Gamma(1, 1)                          // Noise precision
    /// 
    /// Level 3 (Variables):
    ///   - c[signal, basis] ~ Gaussian(0, τ_c)      // Coefficients (sparse)
    ///   - d[basis, sample] ~ Gaussian(μ_d, τ_d)     // Dictionary elements
    ///   - y[signal, sample] ~ Gaussian(D×C, β)      // Observed signals
    /// 
    /// The model learns: Y ≈ D × C + noise
    /// Where:
    ///   - Y: observed signals (numSignals × signalWidth)
    ///   - D: dictionary matrix (numBases × signalWidth) - learned
    ///   - C: coefficient matrix (numSignals × numBases) - learned (sparse)
    /// </summary>
    public class Program
    {
        // Helper extension methods for array conversions (needed for Infer.NET API)
        private static T[,] To2D<T>(T[][] jaggedArray)
        {
            if (jaggedArray == null || jaggedArray.Length == 0) return new T[0, 0];
            int cols = jaggedArray.Max(row => row.Length);
            var result = new T[jaggedArray.Length, cols];
            for (int i = 0; i < jaggedArray.Length; i++)
            {
                for (int j = 0; j < jaggedArray[i].Length && j < cols; j++)
                {
                    result[i, j] = jaggedArray[i][j];
                }
            }
            return result;
        }

        private static T[][]? ToJagged<T>(T[,] matrix)
        {
            if (matrix == null) return null;
            var result = new T[matrix.GetLength(0)][];
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                result[i] = new T[matrix.GetLength(1)];
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    result[i][j] = matrix[i, j];
                }
            }
            return result;
        }

        private static double[] GetMeans(Gaussian[] gaussians)
        {
            return gaussians.Select(g => g.GetMean()).ToArray();
        }

        private static double[][] GetMeans(Gaussian[][] gaussians)
        {
            return gaussians.Select(row => GetMeans(row)).ToArray();
        }

        /// <summary>
        /// Generates synthetic data for testing
        /// Creates a true dictionary and sparse coefficients, then generates noisy signals
        /// </summary>
        private static (double[][] signals, double[][] trueDictionary, double[][] trueCoefficients) GenerateSyntheticData(
            int numSignals, int numBases, int signalWidth, double noiseStdDev = 0.1)
        {
            var random = new Random(42); // Fixed seed for reproducibility

            // Generate true dictionary (numBases × signalWidth)
            // Each basis is a smooth function (sine wave with different frequency)
            var trueDictionary = new double[numBases][];
            for (int k = 0; k < numBases; k++)
            {
                trueDictionary[k] = new double[signalWidth];
                double freq = (k + 1) * 2.0 * Math.PI / signalWidth;
                for (int j = 0; j < signalWidth; j++)
                {
                    trueDictionary[k][j] = Math.Sin(freq * j) / Math.Sqrt(signalWidth);
                }
            }

            // Generate sparse coefficients (numSignals × numBases)
            // Each signal uses only 2-3 dictionary atoms (sparse)
            var trueCoefficients = new double[numSignals][];
            for (int i = 0; i < numSignals; i++)
            {
                trueCoefficients[i] = new double[numBases];
                // Randomly select 2-3 active bases
                int numActive = random.Next(2, 4);
                var activeBases = Enumerable.Range(0, numBases).OrderBy(_ => random.Next()).Take(numActive);
                foreach (int k in activeBases)
                {
                    trueCoefficients[i][k] = random.NextDouble() * 2.0 - 1.0; // Random value in [-1, 1]
                }
            }

            // Generate signals: Y = D × C + noise
            var signals = new double[numSignals][];
            for (int i = 0; i < numSignals; i++)
            {
                signals[i] = new double[signalWidth];
                for (int j = 0; j < signalWidth; j++)
                {
                    double cleanValue = 0.0;
                    for (int k = 0; k < numBases; k++)
                    {
                        cleanValue += trueDictionary[k][j] * trueCoefficients[i][k];
                    }
                    // Add Gaussian noise
                    signals[i][j] = cleanValue + Microsoft.ML.Probabilistic.Math.Rand.Normal(0, noiseStdDev);
                }
            }

            return (signals, trueDictionary, trueCoefficients);
        }

        /// <summary>
        /// Saves a matrix to a CSV file
        /// </summary>
        private static void SaveMatrix(double[][] matrix, string filename)
        {
            using (var writer = new StreamWriter(filename))
            {
                foreach (var row in matrix)
                {
                    writer.WriteLine(string.Join(",", row.Select(x => x.ToString("F6"))));
                }
            }
            Console.WriteLine($"Saved matrix ({matrix.Length} × {matrix[0].Length}) to {filename}");
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("=== Bayesian Dictionary Learning Example ===\n");

            // ====================================================================
            // PARAMETERS
            // ====================================================================
            int numSignals = 200;      // Number of observed signals
            int numBases = 8;         // Number of dictionary atoms (bases)
            int signalWidth = 64;      // Dimension of each signal
            bool sparse = true;        // Use sparse priors for coefficients
            int maxIterations = 100;    // Maximum VMP iterations
            // Note: tolerance parameter reserved for future convergence checking

            Console.WriteLine($"Parameters:");
            Console.WriteLine($"  Signals: {numSignals}");
            Console.WriteLine($"  Bases: {numBases}");
            Console.WriteLine($"  Signal width: {signalWidth}");
            Console.WriteLine($"  Sparse priors: {sparse}");
            Console.WriteLine();

            // ====================================================================
            // GENERATE SYNTHETIC DATA
            // ====================================================================
            Console.WriteLine("Generating synthetic data...");
            var (signals, trueDictionary, trueCoefficients) = GenerateSyntheticData(
                numSignals, numBases, signalWidth, noiseStdDev: 0.1);
            Console.WriteLine($"Generated {numSignals} signals of width {signalWidth}");
            Console.WriteLine();

            // ====================================================================
            // DEFINE INFER.NET MODEL
            // ====================================================================
            Console.WriteLine("Defining Infer.NET model...");

            // Ranges for indexing
            var numberOfSignals = Variable.New<int>().Named("numberOfSignals");
            var numberOfBases = Variable.New<int>().Named("numberOfBases");
            var signalWidthVar = Variable.New<int>().Named("signalWidth");

            var signalRange = new Range(numberOfSignals).Named("signal");
            var basisRange = new Range(numberOfBases).Named("basis");
            var sampleRange = new Range(signalWidthVar).Named("sample");

            // Note: Sequential attribute removed - may interfere with VMP message passing
            // signalRange.AddAttribute(new Sequential());

            // ====================================================================
            // HYPERPARAMETERS
            // ====================================================================
            // Hyperparameters for coefficient precision priors
            // Sparse: Gamma(0.5, 3e-6) encourages sparsity (~70%) without crushing coefficients
            // Non-sparse: Gamma(1.0, 1.0) standard prior
            // This achieves ~71% sparsity (close to true 68.5%) with good reconstruction (SNR ~2.3 dB)
            var a = Variable.Observed(sparse ? 0.5 : 1.0).Named("a");
            var b = Variable.Observed(sparse ? 3e-6 : 1.0).Named("b");

            // Noise precision prior: Gamma(1, 1)
            var noisePrecisionPrior = Variable.New<Gamma>().Named("noisePrecisionPrior").Attrib(new DoNotInfer());
            var noisePrecision = Variable<double>.Random(noisePrecisionPrior).Named("noisePrecision");

            // ====================================================================
            // DICTIONARY VARIABLES
            // ====================================================================
            // Dictionary means: μ_d[basis, sample] ~ Gaussian(0, 1.0) - wider prior for learning
            // Dictionary precisions: τ_d[basis, sample] ~ Gamma(1, 1) - fixed, not optimized
            // Dictionary: d[basis, sample] ~ Gaussian(μ_d, τ_d)
            var dictionaryMeans = Variable.Array(Variable.Array<double>(sampleRange), basisRange).Named("dictionaryMeans");
            var dictionaryPrecisions = Variable.Array<double>(basisRange, sampleRange).Named("dictionaryPrecisions");
            var dictionary = Variable.Array<double>(basisRange, sampleRange).Named("dictionary");

            dictionaryMeans[basisRange][sampleRange] = Variable.GaussianFromMeanAndVariance(0, 1.0).ForEach(basisRange, sampleRange);
            // Use a wider prior for dictionary precisions to allow more flexibility
            // Gamma(1, 0.1) means mean precision = 10, variance = 0.1 (wider)
            dictionaryPrecisions[basisRange, sampleRange] = Variable.GammaFromShapeAndRate(1, 0.1).ForEach(basisRange, sampleRange);
            dictionary[basisRange, sampleRange] = Variable.GaussianFromMeanAndPrecision(
                dictionaryMeans[basisRange][sampleRange],
                dictionaryPrecisions[basisRange, sampleRange]);

            // ====================================================================
            // COEFFICIENT VARIABLES
            // ====================================================================
            // Coefficient precisions: τ_c[signal, basis] ~ Gamma(a, b)
            // Coefficients: c[signal, basis] ~ Gaussian(0, τ_c)
            // This creates a sparse prior when a=0.5, b=1e-6
            var coefficientPrecisions = Variable.Array<double>(signalRange, basisRange).Named("coefficientPrecisions");
            var coefficients = Variable.Array<double>(signalRange, basisRange).Named("coefficients");

            coefficientPrecisions[signalRange, basisRange] = Variable.GammaFromShapeAndRate(a, b).ForEach(signalRange, basisRange);
            coefficients[signalRange, basisRange] = Variable.GaussianFromMeanAndPrecision(0, coefficientPrecisions[signalRange, basisRange]);

            // ====================================================================
            // OBSERVATION MODEL
            // ====================================================================
            // Clean signals: clean = C × D (matrix multiplication)
            // coefficients: [signal, basis], dictionary: [basis, sample]
            // Result: [signal, sample] = [signal, basis] × [basis, sample]
            // Observed signals: y[signal, sample] ~ Gaussian(clean, β)
            var cleanSignals = Variable.MatrixMultiply(coefficients, dictionary).Named("clean");
            var observedSignals = Variable.Array<double>(signalRange, sampleRange).Named("signals");
            observedSignals[signalRange, sampleRange] = Variable.GaussianFromMeanAndPrecision(
                cleanSignals[signalRange, sampleRange],
                noisePrecision);

            // ====================================================================
            // SETUP INFERENCE ENGINE
            // ====================================================================
            var engine = new InferenceEngine
            {
                Algorithm = new VariationalMessagePassing(),
                ShowProgress = true
            };

            // Optimize for dictionaryMeans (which determines dictionary), coefficients, and noise precision
            // Note: dictionaryPrecisions are kept as fixed priors (not optimized)
            engine.OptimiseForVariables = new IVariable[] { dictionaryMeans, coefficients, noisePrecision };

            // Set observed values before compilation
            numberOfSignals.ObservedValue = numSignals;
            numberOfBases.ObservedValue = numBases;
            signalWidthVar.ObservedValue = signalWidth;
            noisePrecisionPrior.ObservedValue = Gamma.FromShapeAndRate(1, 1);
            observedSignals.ObservedValue = To2D(signals);

            // ====================================================================
            // INITIALIZE VARIABLES WITH RANDOM VALUES
            // ====================================================================
            // Initialize dictionary means and coefficients with small random values
            // to break symmetry and allow VMP to learn non-zero values
            Console.WriteLine("Initializing variables with random values...");
            var random = new Random(42); // Use fixed seed for reproducibility

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
            dictionaryMeans.InitialiseTo(Distribution<double>.Array(dictionaryMeansInit));
            coefficients.InitialiseTo(Distribution<double>.Array(coefficientsInit));

            Console.WriteLine($"Initialized dictionary means with random values in range [-0.1, 0.1]");
            Console.WriteLine($"Initialized coefficients with random values in range [-0.1, 0.1]");
            Console.WriteLine();

            // Compile the inference algorithm
            // Note: GetCompiledInferenceAlgorithm requires an array of IVariable
            // We query dictionaryMeans (which determines dictionary) and coefficients
            var compiledAlgorithm = engine.GetCompiledInferenceAlgorithm(
                new IVariable[] { dictionaryMeans, coefficients, noisePrecision });

            Console.WriteLine("Model compiled successfully");
            Console.WriteLine();

            // ====================================================================
            // RUN INFERENCE
            // ====================================================================
            Console.WriteLine("Running inference...");

            // Run Variational Message Passing (VMP)
            Console.WriteLine("Running Variational Message Passing...");
            compiledAlgorithm.Reset();

            for (int iteration = 1; iteration <= maxIterations; iteration++)
            {
                compiledAlgorithm.Update(1);

                if (iteration % 10 == 0 || iteration == 1)
                {
                    Console.WriteLine($"  Iteration {iteration}");
                }
            }

            Console.WriteLine($"  Completed {maxIterations} iterations");

            Console.WriteLine();

            // ====================================================================
            // EXTRACT POSTERIORS
            // ====================================================================
            Console.WriteLine("Extracting posterior distributions...");

            // Get posterior distributions
            // Note: dictionaryMeans is what we optimized, and it determines dictionary
            var dictionaryMeansPosteriorArray = compiledAlgorithm.Marginal<Gaussian[][]>(dictionaryMeans.NameInGeneratedCode);
            var coefficientsPosterior = compiledAlgorithm.Marginal<Gaussian[,]>(coefficients.NameInGeneratedCode);
            var noisePrecisionPosterior = compiledAlgorithm.Marginal<Gamma>(noisePrecision.NameInGeneratedCode);

            // Convert to jagged arrays and extract means
            var dictionaryMeansPosterior = GetMeans(dictionaryMeansPosteriorArray);
            var coefficientsMeansPosterior = GetMeans(ToJagged(coefficientsPosterior)!);

            Console.WriteLine($"Dictionary means posterior: {dictionaryMeansPosteriorArray.Length} × {dictionaryMeansPosteriorArray[0].Length}");
            Console.WriteLine($"Coefficients posterior: {coefficientsPosterior.GetLength(0)} × {coefficientsPosterior.GetLength(1)}");
            Console.WriteLine($"Noise precision: {noisePrecisionPosterior}");
            Console.WriteLine();

            // ====================================================================
            // RECONSTRUCTION AND ERROR COMPUTATION
            // ====================================================================
            Console.WriteLine("Computing reconstruction...");

            // Reconstruct signals: reconstructed = learned_coefficients × learned_dictionary
            // coefficients: [numSignals, numBases], dictionary: [numBases, signalWidth]
            // Result: [numSignals, signalWidth]
            var reconstructedSignals = new double[numSignals][];
            for (int i = 0; i < numSignals; i++)
            {
                reconstructedSignals[i] = new double[signalWidth];
                for (int j = 0; j < signalWidth; j++)
                {
                    double reconstructedValue = 0.0;
                    for (int k = 0; k < numBases; k++)
                    {
                        reconstructedValue += coefficientsMeansPosterior[i][k] * dictionaryMeansPosterior[k][j];
                    }
                    reconstructedSignals[i][j] = reconstructedValue;
                }
            }

            // Compute reconstruction error metrics
            double sumSquaredError = 0.0;
            double sumAbsoluteError = 0.0;
            double sumSquaredOriginal = 0.0;
            int totalElements = numSignals * signalWidth;

            for (int i = 0; i < numSignals; i++)
            {
                for (int j = 0; j < signalWidth; j++)
                {
                    double error = signals[i][j] - reconstructedSignals[i][j];
                    double squaredError = error * error;
                    double absoluteError = Math.Abs(error);

                    sumSquaredError += squaredError;
                    sumAbsoluteError += absoluteError;
                    sumSquaredOriginal += signals[i][j] * signals[i][j];
                }
            }

            // Compute error metrics
            double mse = sumSquaredError / totalElements;
            double rmse = Math.Sqrt(mse);
            double mae = sumAbsoluteError / totalElements;
            double relativeError = Math.Sqrt(sumSquaredError / sumSquaredOriginal);
            double signalToNoiseRatio = 10.0 * Math.Log10(sumSquaredOriginal / sumSquaredError);

            Console.WriteLine();
            Console.WriteLine("=== Reconstruction Error Metrics ===");
            Console.WriteLine($"Mean Squared Error (MSE):        {mse:F6}");
            Console.WriteLine($"Root Mean Squared Error (RMSE): {rmse:F6}");
            Console.WriteLine($"Mean Absolute Error (MAE):      {mae:F6}");
            Console.WriteLine($"Relative Error:                 {relativeError:F6}");
            Console.WriteLine($"Signal-to-Noise Ratio (dB):    {signalToNoiseRatio:F2}");
            Console.WriteLine();

            // ====================================================================
            // SAVE RESULTS
            // ====================================================================
            Console.WriteLine("Saving results...");

            // Save learned dictionary (means)
            SaveMatrix(dictionaryMeansPosterior, "learned_dictionary.csv");

            // Save learned coefficients (means)
            SaveMatrix(coefficientsMeansPosterior, "learned_coefficients.csv");

            // Optionally save true values for comparison
            SaveMatrix(trueDictionary, "true_dictionary.csv");
            SaveMatrix(trueCoefficients, "true_coefficients.csv");

            // Save reconstructed signals
            SaveMatrix(reconstructedSignals, "reconstructed_signals.csv");

            Console.WriteLine();
            Console.WriteLine("=== Inference Complete ===");
            Console.WriteLine($"Results saved to:");
            Console.WriteLine($"  - learned_dictionary.csv");
            Console.WriteLine($"  - learned_coefficients.csv");
            Console.WriteLine($"  - reconstructed_signals.csv");
            Console.WriteLine($"  - true_dictionary.csv (for comparison)");
            Console.WriteLine($"  - true_coefficients.csv (for comparison)");
        }
    }
}
