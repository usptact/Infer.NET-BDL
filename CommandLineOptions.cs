//
// CommandLineOptions.cs
//
// Defines command-line options for the Bayesian Dictionary Learning program
//

using CommandLine;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Command-line options for Bayesian Dictionary Learning
    /// </summary>
    public class CommandLineOptions
    {
        [Option('d', "data", Required = false, HelpText = "Path to CSV file containing signal data (one signal per row). If not specified, synthetic data will be generated.")]
        public string? DataFile { get; set; }

        [Option('s', "signals", Required = false, Default = 200, HelpText = "Number of signals (only used for synthetic data generation).")]
        public int NumSignals { get; set; }

        [Option('b', "bases", Required = true, HelpText = "Number of dictionary atoms (bases) to learn.")]
        public int NumBases { get; set; }

        [Option('w', "width", Required = false, HelpText = "Signal width/dimension. Auto-detected from data file if not specified.")]
        public int? SignalWidth { get; set; }

        [Option("sparse", Required = false, Default = true, HelpText = "Use sparse priors for coefficients (ARD-style sparsity).")]
        public bool UseSparse { get; set; }

        [Option('a', "prior-shape", Required = false, Default = 0.5, HelpText = "Shape parameter (a) for coefficient precision prior Gamma(a, b). Lower values encourage more sparsity.")]
        public double PriorShape { get; set; }

        [Option("prior-rate", Required = false, Default = 3e-6, HelpText = "Rate parameter (b) for coefficient precision prior Gamma(a, b). Higher values encourage sparsity.")]
        public double PriorRate { get; set; }

        [Option('i', "iterations", Required = false, Default = 100, HelpText = "Maximum number of VMP iterations.")]
        public int MaxIterations { get; set; }

        [Option("seed", Required = false, Default = 42, HelpText = "Random seed for reproducibility.")]
        public int Seed { get; set; }

        [Option("noise", Required = false, Default = 0.1, HelpText = "Noise standard deviation (only used for synthetic data generation).")]
        public double NoiseStdDev { get; set; }

        [Option('o', "output-prefix", Required = false, Default = "", HelpText = "Prefix for output filenames (e.g., 'experiment1_').")]
        public string OutputPrefix { get; set; } = "";

        [Option('v', "verbose", Required = false, Default = false, HelpText = "Enable verbose output.")]
        public bool Verbose { get; set; }

        [Option("init-dictionary", Required = false, HelpText = "Path to CSV file with initial dictionary values (numBases × signalWidth). If not specified, random initialization is used.")]
        public string? InitDictionaryFile { get; set; }

        [Option("init-coefficients", Required = false, HelpText = "Path to CSV file with initial coefficient values (numSignals × numBases). If not specified, random initialization is used.")]
        public string? InitCoefficientsFile { get; set; }
    }
}

