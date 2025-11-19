//
// FileManager.cs
//
// Handles file I/O operations for saving and loading matrices
//

using System;
using System.IO;
using System.Linq;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Manages file I/O operations for matrices
    /// </summary>
    public static class FileManager
    {
        /// <summary>
        /// Saves a matrix to a CSV file
        /// </summary>
        /// <param name="matrix">Matrix to save</param>
        /// <param name="filename">Filename to save to</param>
        public static void SaveMatrix(double[][] matrix, string filename)
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
    }
}
