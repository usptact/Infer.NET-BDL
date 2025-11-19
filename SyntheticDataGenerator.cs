//
// SyntheticDataGenerator.cs
//
// Generates synthetic data for testing Bayesian Dictionary Learning
//

using System;
using System.Linq;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Generates synthetic data for testing dictionary learning
    /// </summary>
    public class SyntheticDataGenerator
    {
        private readonly Random _random;

        /// <summary>
        /// Creates a new synthetic data generator with a fixed seed for reproducibility
        /// </summary>
        public SyntheticDataGenerator(int seed = 42)
        {
            _random = new Random(seed);
        }

        /// <summary>
        /// Generates synthetic data for testing
        /// Creates a true dictionary and sparse coefficients, then generates noisy signals
        /// </summary>
        /// <param name="numSignals">Number of signals to generate</param>
        /// <param name="numBases">Number of dictionary atoms (bases)</param>
        /// <param name="signalWidth">Dimension of each signal</param>
        /// <param name="noiseStdDev">Standard deviation of Gaussian noise</param>
        /// <returns>Tuple containing signals, true dictionary, and true coefficients</returns>
        public (double[][] signals, double[][] trueDictionary, double[][] trueCoefficients) Generate(
            int numSignals, int numBases, int signalWidth, double noiseStdDev = 0.1)
        {
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
                int numActive = _random.Next(2, 4);
                var activeBases = Enumerable.Range(0, numBases).OrderBy(_ => _random.Next()).Take(numActive);
                foreach (int k in activeBases)
                {
                    trueCoefficients[i][k] = _random.NextDouble() * 2.0 - 1.0; // Random value in [-1, 1]
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
    }
}
