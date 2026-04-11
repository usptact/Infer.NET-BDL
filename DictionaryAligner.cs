//
// DictionaryAligner.cs
//
// Post-hoc canonicalization and evaluation utilities for learned dictionaries.
// Resolves the three inherent symmetries of dictionary learning:
//   Scale:       each atom d_k has a continuous positive-real scale ambiguity with c_k
//   Sign:        each atom has a ±1 sign ambiguity with c_k
//   Permutation: atom ordering is arbitrary
//

using System;
using System.Linq;

namespace BayesianDictionaryLearning
{
    public static class DictionaryAligner
    {
        /// <summary>
        /// Normalises each atom to unit L2 norm and rescales the corresponding
        /// coefficient column by the same factor, preserving Y = C × D.
        /// Resolves the scale ambiguity.
        /// </summary>
        public static (double[][] dictionary, double[][] coefficients) NormaliseAtoms(
            double[][] dictionary, double[][] coefficients)
        {
            int numBases   = dictionary.Length;
            int numSignals = coefficients.Length;
            var d = DeepCopy(dictionary);
            var c = DeepCopy(coefficients);

            for (int k = 0; k < numBases; k++)
            {
                double norm = Math.Sqrt(d[k].Sum(x => x * x));
                if (norm < 1e-12) continue; // zero atom — leave as-is

                for (int j = 0; j < d[k].Length; j++)
                    d[k][j] /= norm;
                for (int i = 0; i < numSignals; i++)
                    c[i][k] *= norm;
            }
            return (d, c);
        }

        /// <summary>
        /// Flips the sign of each atom (and its coefficient column) so that the
        /// element with the largest absolute value is positive.
        /// Resolves the sign ambiguity.
        /// </summary>
        public static (double[][] dictionary, double[][] coefficients) CanonicalSign(
            double[][] dictionary, double[][] coefficients)
        {
            int numBases   = dictionary.Length;
            int numSignals = coefficients.Length;
            var d = DeepCopy(dictionary);
            var c = DeepCopy(coefficients);

            for (int k = 0; k < numBases; k++)
            {
                // Find the element with the largest absolute value
                int maxIdx = 0;
                double maxAbs = Math.Abs(d[k][0]);
                for (int j = 1; j < d[k].Length; j++)
                {
                    double a = Math.Abs(d[k][j]);
                    if (a > maxAbs) { maxAbs = a; maxIdx = j; }
                }

                if (d[k][maxIdx] < 0.0)
                {
                    for (int j = 0; j < d[k].Length; j++)
                        d[k][j] = -d[k][j];
                    for (int i = 0; i < numSignals; i++)
                        c[i][k] = -c[i][k];
                }
            }
            return (d, c);
        }

        /// <summary>
        /// Sorts atoms by decreasing total activity: Σ_i |c[i][k]|.
        /// Resolves the permutation ambiguity for human-readable output.
        /// </summary>
        public static (double[][] dictionary, double[][] coefficients) SortByActivity(
            double[][] dictionary, double[][] coefficients)
        {
            int numBases   = dictionary.Length;
            int numSignals = coefficients.Length;

            double[] activity = new double[numBases];
            for (int k = 0; k < numBases; k++)
                for (int i = 0; i < numSignals; i++)
                    activity[k] += Math.Abs(coefficients[i][k]);

            int[] order = Enumerable.Range(0, numBases)
                                    .OrderByDescending(k => activity[k])
                                    .ToArray();

            var d = new double[numBases][];
            var c = new double[numSignals][];
            for (int i = 0; i < numSignals; i++) c[i] = new double[numBases];

            for (int newK = 0; newK < numBases; newK++)
            {
                int oldK = order[newK];
                d[newK] = (double[])dictionary[oldK].Clone();
                for (int i = 0; i < numSignals; i++)
                    c[i][newK] = coefficients[i][oldK];
            }
            return (d, c);
        }

        /// <summary>
        /// Computes atom recovery: optimally aligns learned atoms to true atoms using
        /// the Hungarian algorithm on |cosine similarity|, then returns the mean
        /// absolute cosine similarity across matched pairs (1 = perfect recovery).
        /// </summary>
        public static double ComputeAtomRecovery(double[][] learned, double[][] trueDictionary)
        {
            int n = Math.Min(learned.Length, trueDictionary.Length);
            var profit = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    profit[i, j] = Math.Abs(CosineSimilarity(learned[i], trueDictionary[j]));

            int[] assignment = SolveAssignment(profit);
            double total = 0.0;
            for (int i = 0; i < n; i++)
                total += profit[i, assignment[i]];
            return total / n;
        }

        // ----------------------------------------------------------------
        // Private helpers
        // ----------------------------------------------------------------

        private static double CosineSimilarity(double[] a, double[] b)
        {
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                na  += a[i] * a[i];
                nb  += b[i] * b[i];
            }
            double denom = Math.Sqrt(na) * Math.Sqrt(nb);
            return denom < 1e-12 ? 0.0 : dot / denom;
        }

        /// <summary>
        /// Solves the linear assignment problem (maximise total profit) using
        /// the O(n³) Hungarian algorithm (Jonker-Volgenant formulation).
        /// Returns assignment[i] = j meaning row i is assigned to column j.
        /// </summary>
        private static int[] SolveAssignment(double[,] profit)
        {
            int n = profit.GetLength(0);
            const double INF = double.MaxValue / 2;

            double[] u   = new double[n + 1];
            double[] v   = new double[n + 1];
            int[]    p   = new int[n + 1]; // p[j] = row assigned to column j (1-indexed)
            int[]    way = new int[n + 1];

            for (int i = 1; i <= n; i++)
            {
                p[0] = i;
                int j0 = 0;
                double[] minv = Enumerable.Repeat(INF, n + 1).ToArray();
                bool[]   used = new bool[n + 1];

                do
                {
                    used[j0] = true;
                    int    i0    = p[j0];
                    double delta = INF;
                    int    j1    = -1;

                    for (int j = 1; j <= n; j++)
                    {
                        if (used[j]) continue;
                        // Minimise the negated profit (we want maximum assignment)
                        double cur = -profit[i0 - 1, j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }

                    for (int j = 0; j <= n; j++)
                    {
                        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                        else          minv[j]   -= delta;
                    }
                    j0 = j1;
                } while (p[j0] != 0);

                do
                {
                    int j1 = way[j0];
                    p[j0]  = p[j1];
                    j0     = j1;
                } while (j0 != 0);
            }

            // Convert from 1-indexed column-to-row mapping to row-to-column
            int[] result = new int[n];
            for (int j = 1; j <= n; j++)
                if (p[j] > 0) result[p[j] - 1] = j - 1;
            return result;
        }

        private static double[][] DeepCopy(double[][] arr)
        {
            var copy = new double[arr.Length][];
            for (int i = 0; i < arr.Length; i++)
                copy[i] = (double[])arr[i].Clone();
            return copy;
        }
    }
}
