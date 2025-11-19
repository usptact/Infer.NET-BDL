//
// Program.cs
//
// Bayesian Dictionary Learning using Infer.NET
// Main entry point that orchestrates the dictionary learning pipeline
//

using System;
using CommandLine;

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
            // Parse command-line arguments
            Parser.Default.ParseArguments<CommandLineOptions>(args)
                .WithParsed(options => RunDictionaryLearning(options))
                .WithNotParsed(errors => Environment.Exit(1));
        }

        private static void RunDictionaryLearning(CommandLineOptions options)
        {
            Console.WriteLine("=== Bayesian Dictionary Learning ===\n");

            // ====================================================================
            // LOAD OR GENERATE DATA
            // ====================================================================
            double[][] signals;
            double[][]? trueDictionary = null;
            double[][]? trueCoefficients = null;
            int numSignals;
            int signalWidth;

            if (!string.IsNullOrEmpty(options.DataFile))
            {
                // Load data from CSV file
                Console.WriteLine($"Loading data from: {options.DataFile}");
                signals = DataLoader.LoadSignalsFromCsv(options.DataFile);
                (numSignals, signalWidth) = DataLoader.GetDataInfo(signals);
                Console.WriteLine($"Loaded {numSignals} signals of width {signalWidth}");
                
                // Override signal width if specified
                if (options.SignalWidth.HasValue && options.SignalWidth.Value != signalWidth)
                {
                    Console.WriteLine($"Warning: Specified signal width ({options.SignalWidth}) does not match data ({signalWidth}). Using data width.");
                }
            }
            else
            {
                // Generate synthetic data
                Console.WriteLine("Generating synthetic data...");
                numSignals = options.NumSignals;
                signalWidth = options.SignalWidth ?? 64; // Default to 64 if not specified
                
                var dataGenerator = new SyntheticDataGenerator(seed: options.Seed);
                (signals, trueDictionary, trueCoefficients) = dataGenerator.Generate(
                    numSignals, options.NumBases, signalWidth, noiseStdDev: options.NoiseStdDev);
                Console.WriteLine($"Generated {numSignals} signals of width {signalWidth}");
            }
            Console.WriteLine();

            // ====================================================================
            // DISPLAY PARAMETERS
            // ====================================================================
            Console.WriteLine("Parameters:");
            Console.WriteLine($"  Data source: {(string.IsNullOrEmpty(options.DataFile) ? "Synthetic" : options.DataFile)}");
            Console.WriteLine($"  Signals: {numSignals}");
            Console.WriteLine($"  Bases: {options.NumBases}");
            Console.WriteLine($"  Signal width: {signalWidth}");
            Console.WriteLine($"  Sparse priors: {options.UseSparse}");
            Console.WriteLine($"  Prior shape (a): {options.PriorShape}");
            Console.WriteLine($"  Prior rate (b): {options.PriorRate}");
            Console.WriteLine($"  Max iterations: {options.MaxIterations}");
            Console.WriteLine($"  Random seed: {options.Seed}");
            if (options.Verbose)
            {
                Console.WriteLine($"  Verbose mode: enabled");
                Console.WriteLine($"  Output prefix: {(string.IsNullOrEmpty(options.OutputPrefix) ? "(none)" : options.OutputPrefix)}");
            }
            Console.WriteLine();

            // ====================================================================
            // DEFINE AND RUN MODEL
            // ====================================================================
            Console.WriteLine("Defining Infer.NET model...");
            var model = new ModelDefinition(
                sparse: options.UseSparse, 
                priorShape: options.PriorShape, 
                priorRate: options.PriorRate);
            model.SetObservedValues(numSignals, options.NumBases, signalWidth, signals);

            var inference = new ModelInference(model);
            
            // ====================================================================
            // LOAD CUSTOM INITIALIZATION (if specified)
            // ====================================================================
            double[][]? customDictionary = null;
            double[][]? customCoefficients = null;

            if (!string.IsNullOrEmpty(options.InitDictionaryFile))
            {
                Console.WriteLine($"Loading dictionary initialization from: {options.InitDictionaryFile}");
                try
                {
                    customDictionary = DataLoader.LoadInitializationMatrix(
                        options.InitDictionaryFile,
                        expectedRows: options.NumBases,
                        expectedCols: signalWidth,
                        matrixName: "Dictionary");
                    Console.WriteLine($"  Loaded dictionary initialization: {options.NumBases} × {signalWidth}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error loading dictionary initialization: {ex.Message}");
                    Environment.Exit(1);
                }
            }

            if (!string.IsNullOrEmpty(options.InitCoefficientsFile))
            {
                Console.WriteLine($"Loading coefficients initialization from: {options.InitCoefficientsFile}");
                try
                {
                    customCoefficients = DataLoader.LoadInitializationMatrix(
                        options.InitCoefficientsFile,
                        expectedRows: numSignals,
                        expectedCols: options.NumBases,
                        matrixName: "Coefficients");
                    Console.WriteLine($"  Loaded coefficients initialization: {numSignals} × {options.NumBases}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error loading coefficients initialization: {ex.Message}");
                    Environment.Exit(1);
                }
            }
            Console.WriteLine();

            Console.WriteLine("Initializing variables...");
            inference.InitializeVariables(
                numSignals, 
                options.NumBases, 
                signalWidth, 
                seed: options.Seed,
                customDictionary: customDictionary,
                customCoefficients: customCoefficients);
            Console.WriteLine();

            inference.Compile();
            Console.WriteLine();

            Console.WriteLine("Running inference...");
            inference.RunInference(options.MaxIterations);
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
            string prefix = options.OutputPrefix;
            FileManager.SaveMatrix(results.DictionaryMeans, $"{prefix}learned_dictionary.csv");
            FileManager.SaveMatrix(results.CoefficientsMeans, $"{prefix}learned_coefficients.csv");
            FileManager.SaveMatrix(reconstructedSignals, $"{prefix}reconstructed_signals.csv");
            
            // Save ground truth if using synthetic data
            if (trueDictionary != null && trueCoefficients != null)
            {
                FileManager.SaveMatrix(trueDictionary, $"{prefix}true_dictionary.csv");
                FileManager.SaveMatrix(trueCoefficients, $"{prefix}true_coefficients.csv");
            }

            Console.WriteLine();
            Console.WriteLine("=== Inference Complete ===");
            Console.WriteLine($"Results saved to:");
            Console.WriteLine($"  - {prefix}learned_dictionary.csv");
            Console.WriteLine($"  - {prefix}learned_coefficients.csv");
            Console.WriteLine($"  - {prefix}reconstructed_signals.csv");
            if (trueDictionary != null && trueCoefficients != null)
            {
                Console.WriteLine($"  - {prefix}true_dictionary.csv (ground truth)");
                Console.WriteLine($"  - {prefix}true_coefficients.csv (ground truth)");
            }
        }
    }
}
