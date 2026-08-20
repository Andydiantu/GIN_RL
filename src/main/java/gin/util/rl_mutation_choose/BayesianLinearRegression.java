package gin.util.rl_mutation_choose;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

import java.util.Arrays;

/**
 * Bayesian Linear Regression with a Gaussian prior and known noise variance.
 */
public class BayesianLinearRegression implements LinearRegression {
    private final int dim;
    private final double noiseVariance;
    private RealMatrix precision; // A
    private RealVector b; // A μ

    public BayesianLinearRegression(int dim, double priorVariance, double noiseVariance) {
        this(dim, uniformPriorVariances(dim, priorVariance), noiseVariance);
    }

    public BayesianLinearRegression(int dim, double[] priorVariances, double noiseVariance) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive");
        }
        if (noiseVariance <= 0 || Double.isNaN(noiseVariance) || Double.isInfinite(noiseVariance)) {
            throw new IllegalArgumentException("noiseVariance must be finite and positive");
        }
        if (priorVariances == null || priorVariances.length != dim) {
            throw new IllegalArgumentException("priorVariances length must match dim");
        }
        for (double priorVariance : priorVariances) {
            if (priorVariance <= 0 || Double.isNaN(priorVariance) || Double.isInfinite(priorVariance)) {
                throw new IllegalArgumentException("All prior variances must be finite and positive");
            }
        }

        this.dim = dim;
        this.noiseVariance = noiseVariance;
        // initialize A = Σ₀⁻¹
        this.precision = new DiagonalMatrix(invertVariances(priorVariances));
        // initialize b = Σ₀⁻¹ μ₀ = 0
        this.b = new ArrayRealVector(dim);
    }

    /**
     * Incorporate one new observation (x, y).
     */
    public void update(double[] xArray, double y) {
        RealVector x = new ArrayRealVector(xArray);
        // A += (1/σ²) x xᵀ
        RealMatrix outer = x.outerProduct(x).scalarMultiply(1.0 / noiseVariance);
        // Keep the posterior precision matrix dense after observing off-diagonal correlations.
        precision = outer.add(precision);
        // b += (1/σ²) x y
        RealVector xb = x.mapMultiply(y / noiseVariance);
        b = b.add(xb);
    }

    /** @return the posterior mean vector μ = A⁻¹ b */
    public RealVector getPosteriorMean() {
        DecompositionSolver solver = posteriorSolver();
        return solver.solve(b);
    }

    /** @return the posterior covariance matrix Σ = A⁻¹ */
    public RealMatrix getPosteriorCovariance() {
        DecompositionSolver solver = posteriorSolver();
        return solver.getInverse();
    }

    /** @return the posterior mean prediction μᵀx */
    public double predictMean(double[] xArray) {
        RealVector mu = getPosteriorMean();
        RealVector x = new ArrayRealVector(xArray);
        return mu.dotProduct(x);
    }

    /**
     * Sample a weight vector θ ~ N(μ, Σ) and return its prediction θᵀ x.
     */
    public double predictSample(double[] xArray) {
        RealVector mu = getPosteriorMean();
        RealMatrix sigma = getPosteriorCovariance();
        RealVector x = new ArrayRealVector(xArray);
        try {
            MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(mu.toArray(), sigma.getData());
            double prediction = new ArrayRealVector(mvn.sample()).dotProduct(x);
            if (Double.isFinite(prediction)) {
                return prediction;
            }
        } catch (RuntimeException e) {
        }
        return mu.dotProduct(x);
    }

    /** Draw one coefficient vector for reuse across all operators in a decision. */
    public double[] sampleCoefficients() {
        RealVector mu = getPosteriorMean();
        RealMatrix sigma = getPosteriorCovariance();
        try {
            double[] sample = new MultivariateNormalDistribution(mu.toArray(), sigma.getData()).sample();
            for (double value : sample) {
                if (!Double.isFinite(value)) {
                    return mu.toArray();
                }
            }
            return sample;
        } catch (RuntimeException e) {
            return mu.toArray();
        }
    }

    /** Predict theta^T x for an explicitly supplied coefficient vector. */
    public double predict(double[] xArray, double[] coefficients) {
        if (xArray.length != dim || coefficients.length != dim) {
            throw new IllegalArgumentException("Feature/coefficient dimension mismatch");
        }
        return new ArrayRealVector(coefficients).dotProduct(new ArrayRealVector(xArray));
    }

    private DecompositionSolver posteriorSolver() {
        RealMatrix symmetricPrecision = precision.add(precision.transpose()).scalarMultiply(0.5);
        return new CholeskyDecomposition(symmetricPrecision, 1e-10, 1e-15).getSolver();
    }

    private static double[] uniformPriorVariances(int dim, double priorVariance) {
        if (priorVariance <= 0 || Double.isNaN(priorVariance) || Double.isInfinite(priorVariance)) {
            throw new IllegalArgumentException("priorVariance must be finite and positive");
        }
        double[] variances = new double[dim];
        Arrays.fill(variances, priorVariance);
        return variances;
    }

    private static double[] invertVariances(double[] priorVariances) {
        double[] priorPrecisionDiagonal = new double[priorVariances.length];
        for (int i = 0; i < priorVariances.length; i++) {
            priorPrecisionDiagonal[i] = 1.0 / priorVariances[i];
        }
        return priorPrecisionDiagonal;
    }

}
