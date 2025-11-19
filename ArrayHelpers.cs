//
// ArrayHelpers.cs
//
// Helper methods for array conversions and operations
//

using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Provides utility methods for array conversions and operations
    /// </summary>
    public static class ArrayHelpers
    {
        /// <summary>
        /// Converts a jagged array to a 2D array
        /// </summary>
        public static T[,] To2D<T>(T[][] jaggedArray)
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

        /// <summary>
        /// Converts a 2D array to a jagged array
        /// </summary>
        public static T[][]? ToJagged<T>(T[,] matrix)
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

        /// <summary>
        /// Extracts means from an array of Gaussian distributions
        /// </summary>
        public static double[] GetMeans(Gaussian[] gaussians)
        {
            return gaussians.Select(g => g.GetMean()).ToArray();
        }

        /// <summary>
        /// Extracts means from a jagged array of Gaussian distributions
        /// </summary>
        public static double[][] GetMeans(Gaussian[][] gaussians)
        {
            return gaussians.Select(row => GetMeans(row)).ToArray();
        }
    }
}
