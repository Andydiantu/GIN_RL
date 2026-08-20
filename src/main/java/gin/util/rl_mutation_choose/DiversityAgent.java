package gin.util.rl_mutation_choose;

import java.util.HashMap;

public class DiversityAgent {
    private static final double CONTEXT_PRIOR_VARIANCE = 1.0;
    private static final double BAND_BIAS_PRIOR_VARIANCE = 1.0;
    private static final double INTERACTION_PRIOR_VARIANCE = 1.0;

    private final int contextDim;
    private final BanditFeatureEncoder encoder;
    private final BayesianLogisticRegression blr;
    private final int[] bandCount;

    public DiversityAgent(int numBands, int numFeatures) {
        this(numBands, numFeatures, BanditFeatureMapMode.ADDITIVE);
    }

    public DiversityAgent(int numBands, int numFeatures, BanditFeatureMapMode mode) {
        this.contextDim = numFeatures;
        this.encoder = new BanditFeatureEncoder(numFeatures, numBands, mode);
        double[] priorVariances = encoder.buildPriorVariances(
                CONTEXT_PRIOR_VARIANCE,
                BAND_BIAS_PRIOR_VARIANCE,
                INTERACTION_PRIOR_VARIANCE);
        this.blr = new BayesianLogisticRegression(encoder.getEncodedDimension(), priorVariances, 100);
        this.bandCount = new int[numBands];
    }

    public void update(int band, double[] xArray, int y, HashMap<Integer, Integer> mutationCount) {
        blr.update(encoder.encode(xArray, band), y);
        bandCount[band]++;
    }

    public double predict(int band, double[] xArray) {
        if (xArray.length != contextDim) {
            throw new IllegalArgumentException("xArray length must be equal to numFeatures");
        }
        return blr.predictProbSample(encoder.encode(xArray, band));
    }

    public double[] selectBand(double[] xArray, int[] selectedBands, HashMap<Integer, Integer> mutationCount) {
        double[] sampledProb = new double[selectedBands.length];
        double[] sampledCoefficients = blr.sampleCoefficients();
        for (int i = 0; i < selectedBands.length; i++) {
            sampledProb[i] = blr.predictProb(encoder.encode(xArray, selectedBands[i]), sampledCoefficients);
        }
        return sampledProb;
    }

    public double predictMean(int band, double[] xArray) {
        if (xArray.length != contextDim) {
            throw new IllegalArgumentException("xArray length must be equal to numFeatures");
        }
        return blr.predictProbMean(encoder.encode(xArray, band));
    }

    int getEncodedFeatureDimension() {
        return encoder.getEncodedDimension();
    }

    BanditFeatureMapMode getFeatureMapMode() {
        return encoder.getMode();
    }
}
