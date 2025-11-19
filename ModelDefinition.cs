//
// ModelDefinition.cs
//
// Defines the Bayesian Dictionary Learning model structure using Infer.NET
//

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace BayesianDictionaryLearning
{
    /// <summary>
    /// Defines the Bayesian Dictionary Learning model
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
    /// </summary>
    public class ModelDefinition
    {
        // Model variables
        public Variable<int> NumberOfSignals { get; private set; }
        public Variable<int> NumberOfBases { get; private set; }
        public Variable<int> SignalWidth { get; private set; }
        
        public Range SignalRange { get; private set; }
        public Range BasisRange { get; private set; }
        public Range SampleRange { get; private set; }
        
        public Variable<double> A { get; private set; }
        public Variable<double> B { get; private set; }
        
        public Variable<Gamma> NoisePrecisionPrior { get; private set; }
        public Variable<double> NoisePrecision { get; private set; }
        
        public VariableArray<VariableArray<double>, double[][]> DictionaryMeans { get; private set; }
        public VariableArray2D<double> DictionaryPrecisions { get; private set; }
        public VariableArray2D<double> Dictionary { get; private set; }
        
        public VariableArray2D<double> CoefficientPrecisions { get; private set; }
        public VariableArray2D<double> Coefficients { get; private set; }
        
        public VariableArray2D<double> CleanSignals { get; private set; }
        public VariableArray2D<double> ObservedSignals { get; private set; }

        /// <summary>
        /// Creates and defines the Bayesian Dictionary Learning model
        /// </summary>
        /// <param name="sparse">Whether to use sparse priors for coefficients</param>
        /// <param name="priorShape">Shape parameter (a) for coefficient precision prior Gamma(a, b)</param>
        /// <param name="priorRate">Rate parameter (b) for coefficient precision prior Gamma(a, b)</param>
        public ModelDefinition(bool sparse = true, double? priorShape = null, double? priorRate = null)
        {
            // Ranges for indexing
            NumberOfSignals = Variable.New<int>().Named("numberOfSignals");
            NumberOfBases = Variable.New<int>().Named("numberOfBases");
            SignalWidth = Variable.New<int>().Named("signalWidth");

            SignalRange = new Range(NumberOfSignals).Named("signal");
            BasisRange = new Range(NumberOfBases).Named("basis");
            SampleRange = new Range(SignalWidth).Named("sample");

            // Hyperparameters for coefficient precision priors
            // Default sparse: Gamma(0.5, 3e-6) encourages sparsity (~70%)
            // Default non-sparse: Gamma(1.0, 1.0) standard prior
            double a = priorShape ?? (sparse ? 0.5 : 1.0);
            double b = priorRate ?? (sparse ? 3e-6 : 1.0);
            
            A = Variable.Observed(a).Named("a");
            B = Variable.Observed(b).Named("b");

            // Noise precision prior: Gamma(1, 1)
            NoisePrecisionPrior = Variable.New<Gamma>().Named("noisePrecisionPrior").Attrib(new DoNotInfer());
            NoisePrecision = Variable<double>.Random(NoisePrecisionPrior).Named("noisePrecision");

            // Dictionary variables
            DictionaryMeans = Variable.Array(Variable.Array<double>(SampleRange), BasisRange).Named("dictionaryMeans");
            DictionaryPrecisions = Variable.Array<double>(BasisRange, SampleRange).Named("dictionaryPrecisions");
            Dictionary = Variable.Array<double>(BasisRange, SampleRange).Named("dictionary");

            DictionaryMeans[BasisRange][SampleRange] = Variable.GaussianFromMeanAndVariance(0, 1.0).ForEach(BasisRange, SampleRange);
            DictionaryPrecisions[BasisRange, SampleRange] = Variable.GammaFromShapeAndRate(1, 0.1).ForEach(BasisRange, SampleRange);
            Dictionary[BasisRange, SampleRange] = Variable.GaussianFromMeanAndPrecision(
                DictionaryMeans[BasisRange][SampleRange],
                DictionaryPrecisions[BasisRange, SampleRange]);

            // Coefficient variables
            CoefficientPrecisions = Variable.Array<double>(SignalRange, BasisRange).Named("coefficientPrecisions");
            Coefficients = Variable.Array<double>(SignalRange, BasisRange).Named("coefficients");

            CoefficientPrecisions[SignalRange, BasisRange] = Variable.GammaFromShapeAndRate(A, B).ForEach(SignalRange, BasisRange);
            Coefficients[SignalRange, BasisRange] = Variable.GaussianFromMeanAndPrecision(0, CoefficientPrecisions[SignalRange, BasisRange]);

            // Observation model
            CleanSignals = Variable.MatrixMultiply(Coefficients, Dictionary).Named("clean");
            ObservedSignals = Variable.Array<double>(SignalRange, SampleRange).Named("signals");
            ObservedSignals[SignalRange, SampleRange] = Variable.GaussianFromMeanAndPrecision(
                CleanSignals[SignalRange, SampleRange],
                NoisePrecision);
        }

        /// <summary>
        /// Sets the observed values for the model
        /// </summary>
        public void SetObservedValues(int numSignals, int numBases, int signalWidth, double[][] signals)
        {
            NumberOfSignals.ObservedValue = numSignals;
            NumberOfBases.ObservedValue = numBases;
            SignalWidth.ObservedValue = signalWidth;
            NoisePrecisionPrior.ObservedValue = Gamma.FromShapeAndRate(1, 1);
            ObservedSignals.ObservedValue = ArrayHelpers.To2D(signals);
        }
    }
}
