package gin.util.rl_mutation_choose;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.List;

/**
 * Frequentist Logistic GLM-UCB (arm selection index on the logit scale).
 *
 * Fits a regularized logistic regression (penalized MLE):
 * maximize sum log Bernoulli(sigmoid(x^T theta)) - (lambda/2)||theta||^2
 *
 * Uses the frequentist design matrix:
 * V_t = lambda I + sum_{i=1..t} x_i x_i^T
 *
 * And a typical GLM-UCB-style confidence radius:
 * beta_t = sqrt(lambda)*S + (R/kappa)*sqrt( 2 log(1/delta) + d log(1 + t L^2 /
 * lambda) )
 *
 * Arm index:
 * index(x) = x^T thetaHat + beta_t * sqrt( x^T V^{-1} x )
 *
 * Notes:
 * - This is "frequentist GLM-UCB style" (confidence-set driven), not Bayesian
 * Laplace posterior UCB.
 * - For arm selection, return index(x) (sigmoid is monotone and unnecessary for
 * ranking).
 */
public class LogisticUCB implements LogisticRegression {

    private final int dim;

    // Regularization strength (frequentist): lambda > 0
    private final double lambda;

    // Assumed bounds for GLM-UCB confidence radius
    private final double L; // feature norm bound: ||x||_2 <= L (normalize features so L ~ 1)
    private final double S; // parameter norm bound: ||theta*||_2 <= S (your assumed bound)
    private final double delta; // failure probability in confidence bound (e.g., 0.01)
    private final double R; // noise/sub-Gaussian proxy (Bernoulli: conservative choice R=0.5)

    // Scaling for the confidence bound (heuristic adjustment)
    private double betaScale = 0.05;

    public void setBetaScale(double betaScale) {
        this.betaScale = betaScale;
    }

    /* ----------------------- Data storage ----------------------- */
    private final List<double[]> xs = new ArrayList<>();
    private final List<Integer> ys = new ArrayList<>();

    /* ----------------------- Model state ------------------------ */
    private RealVector thetaHat; // regularized MLE estimate
    private RealMatrix V; // design matrix: lambda I + sum x x^T
    private RealMatrix VInv; // inverse of V (for bonus)
    private boolean stale = true; // needs refit/reinverse?

    /**
     * Frequentist GLM-UCB constructor.
     *
     * @param dim    feature dimension
     * @param lambda L2 regularization strength (>0)
     * @param L      bound on ||x||_2 (use 1.0 if you normalize features)
     * @param S      assumed bound on ||theta*||_2
     * @param delta  confidence failure prob (e.g. 0.01 or 1e-3)
     */
    public LogisticUCB(int dim, double lambda, double L, double S, double delta) {
        this(dim, lambda, L, S, delta, 0.5);
    }

    public LogisticUCB(int dim) {
        this(dim, 1.0, 1.0, 3.0, 0.01, 0.5);
    }

    /**
     * Same as above but configurable R.
     */
    public LogisticUCB(int dim, double lambda, double L, double S, double delta, double R) {
        if (dim <= 0)
            throw new IllegalArgumentException("dim must be positive");
        if (lambda <= 0)
            throw new IllegalArgumentException("lambda must be > 0");
        if (L <= 0)
            throw new IllegalArgumentException("L must be > 0");
        if (S <= 0)
            throw new IllegalArgumentException("S must be > 0");
        if (!(delta > 0 && delta < 1))
            throw new IllegalArgumentException("delta must be in (0,1)");
        if (R <= 0)
            throw new IllegalArgumentException("R must be > 0");

        this.dim = dim;
        this.lambda = lambda;
        this.L = L;
        this.S = S;
        this.delta = delta;
        this.R = R;

        this.thetaHat = new ArrayRealVector(dim); // start at 0

        // V_0 = lambda I
        this.V = MatrixUtils.createRealIdentityMatrix(dim).scalarMultiply(lambda);
        this.VInv = MatrixUtils.createRealIdentityMatrix(dim).scalarMultiply(1.0 / lambda);
    }

    /**
     * Add an observation (x, y).
     */
    public void update(double[] xArray, int y) {
        if (xArray.length != dim)
            throw new IllegalArgumentException("Feature dimension mismatch");
        if (y != 0 && y != 1)
            throw new IllegalArgumentException("Label must be 0 or 1");

        double[] xCopy = xArray.clone();
        xs.add(xCopy);
        ys.add(y);

        // Update design matrix V := V + x x^T
        RealVector x = new ArrayRealVector(xCopy);
        RealMatrix outer = x.outerProduct(x);
        V = V.add(outer);

        stale = true;
    }

    /**
     * Arm-selection index on the logit scale:
     * x^T thetaHat + beta_t * sqrt(x^T V^{-1} x)
     */
    public double getUCBIndex(double[] xArray) {
        if (xArray.length != dim)
            throw new IllegalArgumentException("Feature dimension mismatch");
        recomputeIfNeeded();

        RealVector x = new ArrayRealVector(xArray);

        double mean = thetaHat.dotProduct(x);
        double var = x.dotProduct(VInv.operate(x));
        double bonus = betaT() * FastMath.sqrt(Math.max(0.0, var));

        return mean + bonus;
    }

    /**
     * Optional: map index to (0,1). Monotone, so ranking is identical to
     * getUCBIndex().
     */
    public double predictProbSample(double[] xArray) {
        return sigmoid(getUCBIndex(xArray));
    }

    public RealVector getCoefficients() {
        recomputeIfNeeded();
        return thetaHat;
    }

    /* ----------------------- Internals -------------------------- */

    private void recomputeIfNeeded() {
        if (!stale)
            return;

        refitThetaHat(); // regularized logistic regression fit
        recomputeVInv(); // inverse of design matrix for UCB bonus

        stale = false;
    }

    /**
     * Regularized logistic regression via Newton-Raphson (penalized MLE).
     * Objective: maximize log-likelihood - (lambda/2)||theta||^2
     */
    private void refitThetaHat() {
        int n = xs.size();
        if (n == 0) {
            thetaHat = new ArrayRealVector(dim);
            return;
        }

        RealMatrix X = new Array2DRowRealMatrix(n, dim);
        RealVector Y = new ArrayRealVector(n);

        for (int i = 0; i < n; i++) {
            X.setRow(i, xs.get(i));
            Y.setEntry(i, ys.get(i));
        }

        RealMatrix lambdaI = MatrixUtils.createRealIdentityMatrix(dim).scalarMultiply(lambda);

        RealVector theta = thetaHat.copy(); // warm start
        for (int iter = 0; iter < 25; iter++) {
            RealVector Xtheta = X.operate(theta);

            double[] pArr = new double[n];
            double[] wArr = new double[n];

            for (int i = 0; i < n; i++) {
                double p = sigmoid(Xtheta.getEntry(i));
                pArr[i] = p;
                wArr[i] = p * (1.0 - p);
            }

            RealVector p = new ArrayRealVector(pArr);
            RealMatrix W = new DiagonalMatrix(wArr);

            // Gradient: X^T(y - p) - lambda * theta
            RealVector grad = X.transpose().operate(Y.subtract(p)).subtract(lambdaI.operate(theta));

            // Hessian of negative log-posterior (positive definite): X^T W X + lambda I
            RealMatrix H = X.transpose().multiply(W).multiply(X).add(lambdaI);

            RealVector delta;
            try {
                delta = new CholeskyDecomposition(H, 1e-10, 1e-15).getSolver().solve(grad);
            } catch (NonPositiveDefiniteMatrixException e) {
                delta = new LUDecomposition(H).getSolver().solve(grad);
            }

            theta = theta.add(delta);

            if (delta.getNorm() / FastMath.max(1.0, theta.getNorm()) < 1e-6) {
                break;
            }
        }

        thetaHat = theta;
    }

    private void recomputeVInv() {
        try {
            VInv = new CholeskyDecomposition(V, 1e-10, 1e-15).getSolver().getInverse();
        } catch (NonPositiveDefiniteMatrixException e) {
            VInv = new LUDecomposition(V).getSolver().getInverse();
        }
    }

    /**
     * GLM-UCB confidence radius beta_t.
     * Requires a lower bound kappa on sigmoid'(z) over |z| <= B.
     * With ||x||<=L and ||theta*||<=S, one uses B = L*S.
     */
    private double betaT() {
        int t = xs.size();

        double B = L * S;
        double kappa = sigmoidDerivativeLowerBound(B);

        // beta_t = sqrt(lambda)*S + (R/kappa)*sqrt( 2 log(1/delta) + d log(1 + t L^2 /
        // lambda) )
        double term1 = FastMath.sqrt(lambda) * S;
        double inside = 2.0 * FastMath.log(1.0 / delta) + dim * FastMath.log(1.0 + (t * L * L) / lambda);
        double term2 = (R / kappa) * FastMath.sqrt(Math.max(0.0, inside));

        return (term1 + term2) * betaScale;
    }

    private static double sigmoid(double z) {
        // numerically stable sigmoid
        if (z >= 0) {
            return 1.0 / (1.0 + FastMath.exp(-z));
        } else {
            double ez = FastMath.exp(z);
            return ez / (1.0 + ez);
        }
    }

    private static double sigmoidDerivativeLowerBound(double B) {
        // For logistic, sigma'(z)=sigma(z)(1-sigma(z)), symmetric and minimized at
        // |z|=B over [-B,B]
        double s = sigmoid(B);
        double d = s * (1.0 - s);
        // avoid divide-by-zero if B is huge
        return FastMath.max(d, 1e-12);
    }
}
