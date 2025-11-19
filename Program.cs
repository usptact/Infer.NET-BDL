//
// Program.cs
//
// Bayesian Dictionary Learning using Infer.NET
// Main entry point that orchestrates the dictionary learning pipeline
//

using System;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Main program for Bayesian Dictionary Learning
    /// Demonstrates learning sparse representations of signals using a probabilistic approach
    /// </summary>
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== Bayesian Dictionary Learning Example ===\n");

            // ====================================================================
            // PARAMETERS
            // ====================================================================
            int numSignals = 200;      // Number of observed signals
            int numBases = 8;          // Number of dictionary atoms (bases)
            int signalWidth = 64;      // Dimension of each signal
            bool sparse = true;        // Use sparse priors for coefficients
            int maxIterations = 100;   // Maximum VMP iterations

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
            var dataGenerator = new SyntheticDataGenerator(seed: 42);
            var (signals, trueDictionary, trueCoefficients) = dataGenerator.Generate(
                numSignals, numBases, signalWidth, noiseStdDev: 0.1);
            Console.WriteLine($"Generated {numSignals} signals of width {signalWidth}");
            Console.WriteLine();

            // ====================================================================
            // DEFINE AND RUN MODEL
            // ====================================================================
            Console.WriteLine("Defining Infer.NET model...");
            var model = new ModelDefinition(sparse);
            model.SetObservedValues(numSignals, numBases, signalWidth, signals);

            var inference = new ModelInference(model);
            
            Console.WriteLine("Initializing variables with random values...");
            inference.InitializeVariables(numSignals, numBases, signalWidth, seed: 42);
            Console.WriteLine();

            inference.Compile();
            Console.WriteLine();

            Console.WriteLine("Running inference...");
            inference.RunInference(maxIterations);
            Console.WriteLine();

            var results = inference.GetResults();
            Console.WriteLine();

            // ====================================================================
            // RECONSTRUCTION AND ERROR COMPUTATION
            // ====================================================================
            Console.WriteLine("Computing reconstruction...");
            var reconstructedSignals = ErrorMetrics.ReconstructSignals(
                results.CoefficientsMeans, 
                results.DictionaryMeans);

            var metrics = ErrorMetrics.Compute(signals, reconstructedSignals);

            Console.WriteLine();
            Console.WriteLine("=== Reconstruction Error Metrics ===");
            Console.WriteLine($"Mean Squared Error (MSE):        {metrics.MeanSquaredError:F6}");
            Console.WriteLine($"Root Mean Squared Error (RMSE): {metrics.RootMeanSquaredError:F6}");
            Console.WriteLine($"Mean Absolute Error (MAE):      {metrics.MeanAbsoluteError:F6}");
            Console.WriteLine($"Relative Error:                 {metrics.RelativeError:F6}");
            Console.WriteLine($"Signal-to-Noise Ratio (dB):    {metrics.SignalToNoiseRatio:F2}");
            Console.WriteLine();

            // ====================================================================
            // SAVE RESULTS
            // ====================================================================
            Console.WriteLine("Saving results...");
            FileManager.SaveMatrix(results.DictionaryMeans, "learned_dictionary.csv");
            FileManager.SaveMatrix(results.CoefficientsMeans, "learned_coefficients.csv");
            FileManager.SaveMatrix(trueDictionary, "true_dictionary.csv");
            FileManager.SaveMatrix(trueCoefficients, "true_coefficients.csv");
            FileManager.SaveMatrix(reconstructedSignals, "reconstructed_signals.csv");

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
