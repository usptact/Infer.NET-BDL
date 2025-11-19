//
// ErrorMetrics.cs
//
// Computes error metrics for reconstruction quality
//

using System;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Contains computed error metrics for signal reconstruction
    /// </summary>
    public class ReconstructionMetrics
    {
        public double MeanSquaredError { get; set; }
        public double RootMeanSquaredError { get; set; }
        public double MeanAbsoluteError { get; set; }
        public double RelativeError { get; set; }
        public double SignalToNoiseRatio { get; set; }
    }

    /// <summary>
    /// Computes error metrics for signal reconstruction
    /// </summary>
    public static class ErrorMetrics
    {
        /// <summary>
        /// Computes reconstruction error metrics between original and reconstructed signals
        /// </summary>
        /// <param name="originalSignals">Original signals</param>
        /// <param name="reconstructedSignals">Reconstructed signals</param>
        /// <returns>Computed error metrics</returns>
        public static ReconstructionMetrics Compute(double[][] originalSignals, double[][] reconstructedSignals)
        {
            if (originalSignals.Length != reconstructedSignals.Length)
                throw new ArgumentException("Signal arrays must have the same length");

            int numSignals = originalSignals.Length;
            int signalWidth = originalSignals[0].Length;

            double sumSquaredError = 0.0;
            double sumAbsoluteError = 0.0;
            double sumSquaredOriginal = 0.0;
            int totalElements = numSignals * signalWidth;

            for (int i = 0; i < numSignals; i++)
            {
                for (int j = 0; j < signalWidth; j++)
                {
                    double error = originalSignals[i][j] - reconstructedSignals[i][j];
                    double squaredError = error * error;
                    double absoluteError = Math.Abs(error);

                    sumSquaredError += squaredError;
                    sumAbsoluteError += absoluteError;
                    sumSquaredOriginal += originalSignals[i][j] * originalSignals[i][j];
                }
            }

            // Compute error metrics
            double mse = sumSquaredError / totalElements;
            double rmse = Math.Sqrt(mse);
            double mae = sumAbsoluteError / totalElements;
            double relativeError = Math.Sqrt(sumSquaredError / sumSquaredOriginal);
            double signalToNoiseRatio = 10.0 * Math.Log10(sumSquaredOriginal / sumSquaredError);

            return new ReconstructionMetrics
            {
                MeanSquaredError = mse,
                RootMeanSquaredError = rmse,
                MeanAbsoluteError = mae,
                RelativeError = relativeError,
                SignalToNoiseRatio = signalToNoiseRatio
            };
        }

        /// <summary>
        /// Reconstructs signals from learned coefficients and dictionary
        /// </summary>
        /// <param name="coefficients">Coefficient matrix (numSignals × numBases)</param>
        /// <param name="dictionary">Dictionary matrix (numBases × signalWidth)</param>
        /// <returns>Reconstructed signals (numSignals × signalWidth)</returns>
        public static double[][] ReconstructSignals(double[][] coefficients, double[][] dictionary)
        {
            int numSignals = coefficients.Length;
            int numBases = coefficients[0].Length;
            int signalWidth = dictionary[0].Length;

            var reconstructedSignals = new double[numSignals][];
            for (int i = 0; i < numSignals; i++)
            {
                reconstructedSignals[i] = new double[signalWidth];
                for (int j = 0; j < signalWidth; j++)
                {
                    double reconstructedValue = 0.0;
                    for (int k = 0; k < numBases; k++)
                    {
                        reconstructedValue += coefficients[i][k] * dictionary[k][j];
                    }
                    reconstructedSignals[i][j] = reconstructedValue;
                }
            }

            return reconstructedSignals;
        }
    }
}
