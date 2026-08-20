package gin.util.rl_mutation_choose;

public class FitnessAgent {
    private static final double CONTEXT_PRIOR_VARIANCE = 5.0;
    private static final double BAND_BIAS_PRIOR_VARIANCE = 5.0;
    private static final double INTERACTION_PRIOR_VARIANCE = 5.0;
    private static final double NOISE_VARIANCE = 5.0;

    private final int contextDim;
    private final BanditFeatureEncoder encoder;
    private final BayesianLinearRegression blr;
    private final int[] bandCount;

    public FitnessAgent(int numBands, int numFeatures) {
        this(numBands, numFeatures, BanditFeatureMapMode.ADDITIVE);
    }

    public FitnessAgent(int numBands, int numFeatures, BanditFeatureMapMode mode) {
        this.contextDim = numFeatures;
        this.encoder = new BanditFeatureEncoder(numFeatures, numBands, mode);
        double[] priorVariances = encoder.buildPriorVariances(
                CONTEXT_PRIOR_VARIANCE,
                BAND_BIAS_PRIOR_VARIANCE,
                INTERACTION_PRIOR_VARIANCE);
        this.blr = new BayesianLinearRegression(encoder.getEncodedDimension(), priorVariances, NOISE_VARIANCE);
        this.bandCount = new int[numBands];
    }

    public double update(int band, double[] contextArray, double normalizedFitnessImprovement) {
        if (contextArray.length != contextDim) {
            throw new IllegalArgumentException("contextArray length must be equal to numFeatures");
        }
        if (!Double.isFinite(normalizedFitnessImprovement) || normalizedFitnessImprovement < 0.0) {
            throw new IllegalArgumentException("normalizedFitnessImprovement must be finite and non-negative");
        }
        double[] contextArrayWithBand = encoder.encode(contextArray, band);
        blr.update(contextArrayWithBand, normalizedFitnessImprovement);
        bandCount[band]++;
        return normalizedFitnessImprovement;
    }

    public double predict(int band, double[] contextArray) {
        if (contextArray.length != contextDim) {
            throw new IllegalArgumentException("contextArray length must be equal to numFeatures");
        }
        return blr.predictSample(encoder.encode(contextArray, band));
    }

    public double predictMean(int band, double[] contextArray) {
        if (contextArray.length != contextDim) {
            throw new IllegalArgumentException("contextArray length must be equal to numFeatures");
        }
        return blr.predictMean(encoder.encode(contextArray, band));
    }

    public double[] selectBand(double[] contextArray, int[] selectedBands) {
        double[] sampledFitnessImprovement = new double[selectedBands.length];
        double[] sampledCoefficients = blr.sampleCoefficients();
        for (int i = 0; i < selectedBands.length; i++) {
            sampledFitnessImprovement[i] = blr.predict(
                    encoder.encode(contextArray, selectedBands[i]), sampledCoefficients);
        }
        return sampledFitnessImprovement;
    }

    int getEncodedFeatureDimension() {
        return encoder.getEncodedDimension();
    }

    BanditFeatureMapMode getFeatureMapMode() {
        return encoder.getMode();
    }
}
