package gin.util.rl_mutation_choose;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Bayesian Logistic Regression with a Gaussian prior using the Laplace
 * approximation.
 *
 * <p>
 * A zero-mean diagonal Gaussian prior is assumed.
 * After observing data \((x_i, y_i)\) with \(y_i \in \{0,1\}\), the posterior
 * is approximated by
 * a Gaussian \(\mathcal{N}(\hat\theta, \Sigma)\) obtained from a second‑order
 * Taylor expansion of
 * the log‑posterior around the MAP estimate \(\hat\theta\).
 * </p>
 *
 * <p>
 * This implementation is suitable for small/medium‑sized datasets (up to a few
 * thousand points).
 * For larger problems consider stochastic variational inference.
 * </p>
 */
public class BayesianLogisticRegression implements LogisticRegression {
    /** Dimensionality of feature vectors. */
    private final int dim;
    /** Prior precision matrix \(\Lambda = \Sigma_0^{-1}\). */
    private final RealMatrix priorPrecision;
    /** Prior covariance matrix \(\Sigma_0\). */
    private final RealMatrix priorCovariance;
    /** Maximum history size for sliding window (-1 for unlimited). */
    private final int maxHistory;

    /*
     * ----------------------- Data storage (small data assumption)
     * -----------------------
     */
    private final List<double[]> xs = new ArrayList<>();
    private final List<Integer> ys = new ArrayList<>();

    /*
     * ----------------------- Posterior (Laplace) parameters
     * -----------------------------
     */
    private RealVector posteriorMean; // \hat{\theta}
    private RealMatrix posteriorCov; // \Sigma

    private boolean stale = true; // true ⇒ posterior needs recomputation

    /*
     * Cached sampler from the current Laplace posterior (invalidated when stale =
     * true).
     */
    private transient MultivariateNormalDistribution sampler = null;

    /**
     * @param dim           dimensionality d of the feature vector \(x \in
     *                      \mathbb{R}^d\)
     * @param priorVariance prior variance \(\sigma_0^2\) of the isotropic Gaussian
     *                      prior
     */
    public BayesianLogisticRegression(int dim, double priorVariance) {
        this(dim, priorVariance, -1);
    }

    /**
     * @param dim            dimensionality d of the feature vector \(x \in
     *                       \mathbb{R}^d\)
     * @param priorVariances per-dimension prior variances
     */
    public BayesianLogisticRegression(int dim, double[] priorVariances) {
        this(dim, priorVariances, -1);
    }

    /**
     * @param dim           dimensionality d of the feature vector \(x \in
     *                      \mathbb{R}^d\)
     * @param priorVariance prior variance \(\sigma_0^2\) of the isotropic Gaussian
     *                      prior
     * @param maxHistory    maximum number of data points to keep in the sliding
     *                      window (-1 for unlimited)
     */
    public BayesianLogisticRegression(int dim, double priorVariance, int maxHistory) {
        this(dim, uniformPriorVariances(dim, priorVariance), maxHistory);
    }

    /**
     * @param dim            dimensionality d of the feature vector \(x \in
     *                       \mathbb{R}^d\)
     * @param priorVariances per-dimension prior variances
     * @param maxHistory     maximum number of data points to keep in the sliding
     *                       window (-1 for unlimited)
     */
    public BayesianLogisticRegression(int dim, double[] priorVariances, int maxHistory) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive");
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
        this.maxHistory = maxHistory;
        this.priorPrecision = new DiagonalMatrix(invertVariances(priorVariances));
        this.priorCovariance = new DiagonalMatrix(priorVariances.clone());
        this.posteriorMean = new ArrayRealVector(dim); // initialised at 0
        this.posteriorCov = priorCovariance.copy();
    }

    /*
     * =============================================================================
     * ======
     * Public API
     * =============================================================================
     * ======
     */

    /**
     * Add a single labelled observation and mark the posterior as stale.
     *
     * @param xArray feature vector (length = {@code dim})
     * @param y      label in {0,1}
     */
    public void update(double[] xArray, int y) {
        if (xArray.length != dim)
            throw new IllegalArgumentException("Feature dimension mismatch");
        if (y != 0 && y != 1)
            throw new IllegalArgumentException("Label must be 0 or 1");

        xs.add(xArray.clone());
        ys.add(y);

        if (maxHistory > 0 && xs.size() > maxHistory) {
            xs.remove(0);
            ys.remove(0);
        }

        stale = true;
        sampler = null; // invalidate cached sampler
    }

    /** Returns the posterior mean vector (MAP parameters). */
    public RealVector getPosteriorMean() {
        recomputePosteriorIfNeeded();
        return posteriorMean;
    }

    /** Returns the posterior covariance matrix from the Laplace approximation. */
    public RealMatrix getPosteriorCovariance() {
        recomputePosteriorIfNeeded();
        return posteriorCov;
    }

    /** Predicts the probability P(y = 1 | x) using the posterior mean (MAP). */
    public double predictProbMean(double[] xArray) {
        if (xArray.length != dim)
            throw new IllegalArgumentException("Feature dimension mismatch");
        RealVector theta = getPosteriorMean();
        return sigmoid(theta.dotProduct(new ArrayRealVector(xArray)));
    }

    /**
     * Draws a parameter sample from the approximate posterior and returns
     * the corresponding predictive probability.
     */
    public double predictProbSample(double[] xArray) {
        if (xArray.length != dim)
            throw new IllegalArgumentException("Feature dimension mismatch");
        recomputePosteriorIfNeeded();
        try {
            if (sampler == null) {
                sampler = new MultivariateNormalDistribution(posteriorMean.toArray(), posteriorCov.getData());
            }
            double score = new ArrayRealVector(sampler.sample()).dotProduct(new ArrayRealVector(xArray));
            if (Double.isFinite(score)) {
                return sigmoid(score);
            }
        } catch (RuntimeException e) {
        }
        return sigmoid(posteriorMean.dotProduct(new ArrayRealVector(xArray)));
    }

    /**
     * Draw one coefficient vector from the current Laplace posterior. The caller
     * can reuse this vector to score every operator in a Thompson-sampling
     * decision, as required by contextual Thompson Sampling.
     */
    public double[] sampleCoefficients() {
        recomputePosteriorIfNeeded();
        try {
            if (sampler == null) {
                sampler = new MultivariateNormalDistribution(posteriorMean.toArray(), posteriorCov.getData());
            }
            double[] sample = sampler.sample();
            for (double value : sample) {
                if (!Double.isFinite(value)) {
                    return posteriorMean.toArray();
                }
            }
            return sample;
        } catch (RuntimeException e) {
            return posteriorMean.toArray();
        }
    }

    /** Predict P(y=1|x,theta) for an explicitly supplied coefficient vector. */
    public double predictProb(double[] xArray, double[] coefficients) {
        if (xArray.length != dim || coefficients.length != dim) {
            throw new IllegalArgumentException("Feature/coefficient dimension mismatch");
        }
        return sigmoid(new ArrayRealVector(coefficients).dotProduct(new ArrayRealVector(xArray)));
    }

    /*
     * =============================================================================
     * ======
     * Internal: posterior recomputation
     * =============================================================================
     * ======
     */

    /** Ensure the Laplace posterior parameters are up‑to‑date. */
    private void recomputePosteriorIfNeeded() {
        if (!stale)
            return;

        int n = xs.size();
        if (n == 0) {
            // No data ⇒ posterior equals the prior.
            this.posteriorMean = new ArrayRealVector(dim);
            this.posteriorCov = priorCovariance.copy();
            this.stale = false;
            return;
        }

        /*
         * ---------- Build design matrix X (n × d) and target vector y (n) ------------
         */
        RealMatrix X = new Array2DRowRealMatrix(n, dim);
        RealVector Y = new ArrayRealVector(n);
        for (int i = 0; i < n; i++) {
            X.setRow(i, xs.get(i));
            Y.setEntry(i, ys.get(i));
        }

        /*
         * ---------- Newton–Raphson to find MAP ---------------------------------------
         */
        RealVector theta = posteriorMean.copy(); // warm‑start from previous MAP (or 0 initially)
        for (int iter = 0; iter < 25; iter++) {
            RealVector Xtheta = X.operate(theta);

            double[] pArr = new double[n];
            for (int i = 0; i < n; i++)
                pArr[i] = sigmoid(Xtheta.getEntry(i));
            RealVector p = new ArrayRealVector(pArr);

            // Gradient of log‑posterior: Xᵀ(y - p) - Λ θ
            RealVector grad = X.transpose().operate(Y.subtract(p)).subtract(priorPrecision.operate(theta));

            // Hessian (negative log‑posterior): Xᵀ W X + Λ, where W = diag(p*(1-p))
            double[] wDiag = new double[n];
            for (int i = 0; i < n; i++)
                wDiag[i] = pArr[i] * (1.0 - pArr[i]);
            RealMatrix W = new DiagonalMatrix(wDiag);
            RealMatrix H = X.transpose().multiply(W).multiply(X).add(priorPrecision);

            // Enforce symmetry to mitigate floating-point numerical drift
            H = H.add(H.transpose()).scalarMultiply(0.5);

            // H is SPD ⇒ use Cholesky for stability
            DecompositionSolver solver = new CholeskyDecomposition(H, 1e-10, 1e-15).getSolver();
            RealVector delta = solver.solve(grad);
            theta = theta.add(delta);

            if (delta.getNorm() / FastMath.max(1.0, theta.getNorm()) < 1e-6)
                break;
        }

        /*
         * ---------- Covariance: inverse Hessian at MAP ------------------------------
         */
        RealVector XthetaFinal = X.operate(theta);
        double[] wDiagFinal = new double[n];
        for (int i = 0; i < n; i++) {
            double p = sigmoid(XthetaFinal.getEntry(i));
            wDiagFinal[i] = p * (1.0 - p);
        }
        RealMatrix Hfinal = X.transpose().multiply(new DiagonalMatrix(wDiagFinal)).multiply(X).add(priorPrecision);

        // Enforce symmetry to mitigate floating-point numerical drift
        Hfinal = Hfinal.add(Hfinal.transpose()).scalarMultiply(0.5);

        RealMatrix cov = new CholeskyDecomposition(Hfinal, 1e-10, 1e-15).getSolver().getInverse();

        // Enforce symmetry to mitigate numerical drift
        cov = cov.add(cov.transpose()).scalarMultiply(0.5);

        /*
         * ---------- Store & mark up‑to‑date -----------------------------------------
         */
        this.posteriorMean = theta;
        this.posteriorCov = cov;
        this.stale = false;
        this.sampler = null; // will be recreated lazily on next sample request
    }

    /*
     * =============================================================================
     * ======
     * Helpers
     * =============================================================================
     * ======
     */

    /** Numerically stable sigmoid function σ(z) = 1/(1+e^{−z}). */
    private static double sigmoid(double z) {
        if (z >= 0) {
            double ez = FastMath.exp(-z);
            return 1.0 / (1.0 + ez); // safe when z ≫ 0
        } else {
            double ez = FastMath.exp(z);
            return ez / (1.0 + ez); // safe when z ≪ 0
        }
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
