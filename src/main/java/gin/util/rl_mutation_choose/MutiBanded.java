package gin.util.rl_mutation_choose;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.pmw.tinylog.Logger;

import java.util.Arrays;
import java.util.HashMap;

public class MutiBanded {
    private static final double CONTEXT_PRIOR_VARIANCE = 1.0;
    private static final double BAND_BIAS_PRIOR_VARIANCE = 1.0;
    private static final double INTERACTION_PRIOR_VARIANCE = 1.0;

    private final int numBands;
    private final BanditFeatureEncoder encoder;
    private final BayesianLogisticRegression blr;
    private final int[] bandCounts;
    // TODO: Implemented a decay topK scheme
    private int topK;

    public MutiBanded(int numBands, int numFeatures, int topK) {
        this(numBands, numFeatures, topK, BanditFeatureMapMode.ADDITIVE);
    }

    public MutiBanded(int numBands, int numFeatures, int topK, BanditFeatureMapMode mode) {
        this.numBands = numBands;
        this.encoder = new BanditFeatureEncoder(numFeatures, numBands, mode);
        double[] priorVariances = encoder.buildPriorVariances(
                CONTEXT_PRIOR_VARIANCE,
                BAND_BIAS_PRIOR_VARIANCE,
                INTERACTION_PRIOR_VARIANCE);
        this.blr = new BayesianLogisticRegression(encoder.getEncodedDimension(), priorVariances);
        this.bandCounts = new int[numBands];
        this.topK = topK;
    }

    public void update(int band, double[] xArray, int label, HashMap<Integer, Integer> mutationCount) {
        blr.update(encoder.encode(xArray, band), label);
        bandCounts[band]++;
    }

    public double predictProbMean(int band, double[] xArray) {
        return blr.predictProbMean(encoder.encode(xArray, band));
    }

    public double[] selectBand(double[] xArray, HashMap<Integer, Integer> mutationCount) {
        double[] sampledProbabilities = new double[numBands];
        double[] sampledCoefficients = blr.sampleCoefficients();
        for (int i = 0; i < numBands; i++) {
            double[] encoded = encoder.encode(xArray, i);
            sampledProbabilities[i] = blr.predictProb(encoded, sampledCoefficients);

            Logger.info("The contextual vector is" + Arrays.toString(encoded));
        }

        Logger.info("The sampled validity probabilities are " + Arrays.toString(sampledProbabilities));
        return sampledProbabilities;
    }

    public void setTopK(int topK) {
        this.topK = Math.max(1, topK);
    }

    public int getTopK() {
        return topK;
    }

    public int getBandCounts(int band) {
        return bandCounts[band];
    }

    public String getBandCountsString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < numBands; i++) {
            sb.append("Band ").append(i).append(": ").append(bandCounts[i]).append("\n");
        }
        return sb.toString();
    }

    /**
     * Print the posterior mean (weight) and standard deviation for each feature.
     */
    public void logFeatureWeights() {
        RealVector mean = blr.getPosteriorMean();
        RealMatrix covariance = blr.getPosteriorCovariance();

        Logger.info("=== Multi-Armed Bandit Feature Weights ===");
        Logger.info("Feature map mode: " + encoder.getMode());
        Logger.info("--- Context Features (Shared) ---");
        for (int i = 0; i < encoder.getContextDim(); i++) {
            int idx = encoder.getContextOffset() + i;
            double w = mean.getEntry(idx);
            double std = Math.sqrt(covariance.getEntry(idx, idx));
            Logger.info(String.format("Feature %d: %.4f +/- %.4f", i, w, std));
        }

        Logger.info("mean size " + mean.getDimension());
        Logger.info("covariance size " + covariance.getColumnDimension());
        Logger.info("--- Band Bias Weights ---");
        for (int i = 0; i < numBands; i++) {
            int bandIdx = encoder.getBandBiasOffset() + i;
            double w = mean.getEntry(bandIdx);
            double std = Math.sqrt(covariance.getEntry(bandIdx, bandIdx));
            Logger.info(String.format("Band %d: %.4f +/- %.4f", i, w, std));
        }

        if (encoder.hasInteractionBlock()) {
            Logger.info("--- Interaction Weights (Band-Specific) ---");
            for (int band = 0; band < numBands; band++) {
                Logger.info(String.format("Band %d interactions:", band));
                int bandOffset = encoder.getInteractionOffset() + (band * encoder.getContextDim());
                for (int feature = 0; feature < encoder.getContextDim(); feature++) {
                    int idx = bandOffset + feature;
                    double w = mean.getEntry(idx);
                    double std = Math.sqrt(covariance.getEntry(idx, idx));
                    Logger.info(String.format("Interaction %d: %.4f +/- %.4f", feature, w, std));
                }
            }
        }

        Logger.info("==========================================");
    }

    int getEncodedFeatureDimension() {
        return encoder.getEncodedDimension();
    }

    BanditFeatureMapMode getFeatureMapMode() {
        return encoder.getMode();
    }
}
