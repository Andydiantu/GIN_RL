package gin.util.rl_mutation_choose;

/**
 * Optimized LinUCB using Sherman-Morrison updates.
 * Complexity per update: O(d^2)
 * Complexity per prediction: O(d^2)
 * No explicit matrix inversion required.
 */
public class LinUCB implements LinearRegression {
    private final int dim;
    private final double alpha;

    // We store A_inv directly (inverse of the design matrix)
    private double[][] AInv;
    private double[] b; // Reward vector

    public LinUCB(int dim, double alpha, double lambda) {
        this.dim = dim;
        this.alpha = alpha;

        // Initialize A_inv = (1/lambda) * I
        this.AInv = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            AInv[i][i] = 1.0 / lambda;
        }

        this.b = new double[dim];
    }

    /**
     * Efficiently updates AInv using Sherman-Morrison formula:
     * AInv_new = AInv - (AInv * x * x^T * AInv) / (1 + x^T * AInv * x)
     */
    public void update(double[] x, double reward) {
        // 1. Compute v = AInv * x
        double[] v = matrixVectorMultiply(AInv, x);

        // 2. Compute scalar factor = 1 / (1 + x^T * v)
        double denominator = 1.0 + dotProduct(x, v);

        // 3. Update AInv: AInv -= (v * v^T) / denominator
        // We do this element-wise to avoid creating large temp matrices
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                AInv[i][j] -= (v[i] * v[j]) / denominator;
            }
        }

        // 4. Update b: b += reward * x
        for (int i = 0; i < dim; i++) {
            b[i] += reward * x[i];
        }
    }

    public double predictSample(double[] x) {
        // 1. Compute theta = AInv * b
        double[] theta = matrixVectorMultiply(AInv, b);

        // 2. Mean = theta^T * x
        double mean = dotProduct(theta, x);

        // 3. Variance = x^T * AInv * x
        // We already have generic matrix multiplication, but we can optimize
        // since we just need the scalar quadratic form.
        double[] v = matrixVectorMultiply(AInv, x);
        double variance = dotProduct(x, v);

        // 4. UCB = Mean + alpha * sqrt(Variance)
        double confidence = Math.sqrt(Math.max(0, variance)); // max(0) guards against tiny negative floats
        return mean + (alpha * confidence);
    }

    // --- Helpers ---

    private double[] matrixVectorMultiply(double[][] M, double[] v) {
        double[] result = new double[dim];
        for (int i = 0; i < dim; i++) {
            double sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += M[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    private double dotProduct(double[] v1, double[] v2) {
        double sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += v1[i] * v2[i];
        }
        return sum;
    }
}