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
    ///   - a, b: Shape and rate for coefficient precision priors (sparse: a=0.5, b=3e-6)
    ///   - Gamma(2,1): Prior for per-atom dictionary precisions
    ///   - Gamma(1,1): Prior for noise precision
    ///
    /// Level 2 (Precisions):
    ///   - τ_c[signal, basis] ~ Gamma(a, b)          // Coefficient precisions (per element, ARD)
    ///   - τ_d[basis]         ~ Gamma(2, 1/W)        // Dictionary precisions (per atom; E[1/τ]=2W matches unit-norm atom variance)
    ///   - β                  ~ Gamma(1, 1)          // Noise precision
    ///
    /// Level 3 (Variables):
    ///   - c[signal, basis]   ~ Gaussian(0, τ_c[signal, basis])   // Coefficients (sparse via ARD)
    ///   - d[basis, sample]   ~ Gaussian(0, τ_d[basis])           // Dictionary elements
    ///   - y[signal, sample]  ~ Gaussian((C×D)[signal,sample], β) // Observed signals
    ///
    /// The model learns: Y ≈ C × D + noise
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
        
        public Variable<Gamma> NoisePrecisionPrior { get; private set; }
        public Variable<double> NoisePrecision { get; private set; }

        public Variable<Gamma> DictionaryPrecisionPrior { get; private set; }
        
        public VariableArray<double> DictionaryPrecisions { get; private set; }
        public VariableArray2D<double> Dictionary { get; private set; }
        
        public VariableArray2D<double> CoefficientPrecisions { get; private set; }
        public VariableArray2D<double> Coefficients { get; private set; }
        
        public VariableArray2D<double> CleanSignals { get; private set; }
        public VariableArray2D<double> ObservedSignals { get; private set; }

        /// <summary>
        /// Creates and defines the Bayesian Dictionary Learning model
        /// </summary>
        /// <param name="priorShape">Shape parameter (a) for coefficient precision prior Gamma(a, b)</param>
        /// <param name="priorRate">Rate parameter (b) for coefficient precision prior Gamma(a, b)</param>
        public ModelDefinition(double priorShape, double priorRate)
        {
            // Ranges for indexing
            NumberOfSignals = Variable.New<int>().Named("numberOfSignals");
            NumberOfBases = Variable.New<int>().Named("numberOfBases");
            SignalWidth = Variable.New<int>().Named("signalWidth");

            SignalRange = new Range(NumberOfSignals).Named("signal");
            BasisRange = new Range(NumberOfBases).Named("basis");
            SampleRange = new Range(SignalWidth).Named("sample");

            // Noise precision prior: Gamma(1, 1)
            NoisePrecisionPrior = Variable.New<Gamma>().Named("noisePrecisionPrior").Attrib(new DoNotInfer());
            NoisePrecision = Variable<double>.Random(NoisePrecisionPrior).Named("noisePrecision");

            // Dictionary variables
            // Per-atom precision: Gamma(2, 1) has mean=2 and finite E[1/τ]=1, avoiding the
            // pathological infinite-variance issue of Gamma(1, 1). One precision per atom
            // (not per element) so that each atom is scaled uniformly — per-element precision
            // would create incoherent "patchwork" atoms where individual dimensions are
            // independently shrunk.
            DictionaryPrecisions = Variable.Array<double>(BasisRange).Named("dictionaryPrecisions");
            Dictionary = Variable.Array<double>(BasisRange, SampleRange).Named("dictionary");

            DictionaryPrecisionPrior = Variable.New<Gamma>().Named("dictionaryPrecisionPrior").Attrib(new DoNotInfer());
            DictionaryPrecisions[BasisRange] = Variable<double>.Random(DictionaryPrecisionPrior).ForEach(BasisRange);
            using (Variable.ForEach(BasisRange))
            {
                Dictionary[BasisRange, SampleRange] = Variable.GaussianFromMeanAndPrecision(
                    0.0, DictionaryPrecisions[BasisRange]).ForEach(SampleRange);
            }

            // Coefficient variables
            CoefficientPrecisions = Variable.Array<double>(SignalRange, BasisRange).Named("coefficientPrecisions");
            Coefficients = Variable.Array<double>(SignalRange, BasisRange).Named("coefficients");

            CoefficientPrecisions[SignalRange, BasisRange] = Variable.GammaFromShapeAndRate(priorShape, priorRate).ForEach(SignalRange, BasisRange);
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
            // Gamma(2, 1/signalWidth): shape > 1 keeps E[1/τ] finite; rate = 1/W so that
            // E[1/τ] = shape/rate = 2W matches atom element variance ~1/W for unit-norm atoms.
            DictionaryPrecisionPrior.ObservedValue = Gamma.FromShapeAndRate(2, 1.0 / signalWidth);
            ObservedSignals.ObservedValue = ArrayHelpers.To2D(signals);
        }
    }
}
