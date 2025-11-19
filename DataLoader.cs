//
// DataLoader.cs
//
// Handles loading signal data from CSV files
//

using System;
using System.IO;
using System.Linq;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Loads signal data from CSV files
    /// </summary>
    public static class DataLoader
    {
        /// <summary>
        /// Loads signal data from a CSV file
        /// Each row represents one signal, with comma-separated values
        /// </summary>
        /// <param name="filePath">Path to the CSV file</param>
        /// <returns>Jagged array of signals</returns>
        public static double[][] LoadSignalsFromCsv(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Data file not found: {filePath}");

            var lines = File.ReadAllLines(filePath);
            if (lines.Length == 0)
                throw new InvalidDataException("Data file is empty");

            var signals = new double[lines.Length][];
            
            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].Trim();
                if (string.IsNullOrWhiteSpace(line))
                    throw new InvalidDataException($"Empty line found at row {i + 1}");

                var values = line.Split(',')
                    .Select(v => v.Trim())
                    .Where(v => !string.IsNullOrWhiteSpace(v))
                    .ToArray();

                if (values.Length == 0)
                    throw new InvalidDataException($"No values found in row {i + 1}");

                signals[i] = new double[values.Length];
                for (int j = 0; j < values.Length; j++)
                {
                    if (!double.TryParse(values[j], out signals[i][j]))
                        throw new InvalidDataException($"Invalid number '{values[j]}' at row {i + 1}, column {j + 1}");
                }
            }

            // Validate that all signals have the same width
            int signalWidth = signals[0].Length;
            for (int i = 1; i < signals.Length; i++)
            {
                if (signals[i].Length != signalWidth)
                    throw new InvalidDataException($"Inconsistent signal width: row 1 has {signalWidth} values, but row {i + 1} has {signals[i].Length} values");
            }

            return signals;
        }

        /// <summary>
        /// Gets information about the loaded data
        /// </summary>
        public static (int numSignals, int signalWidth) GetDataInfo(double[][] signals)
        {
            if (signals == null || signals.Length == 0)
                throw new ArgumentException("Signals array is empty");

            return (signals.Length, signals[0].Length);
        }
    }
}

