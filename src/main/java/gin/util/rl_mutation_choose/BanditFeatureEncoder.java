package gin.util.rl_mutation_choose;

public class BanditFeatureEncoder {
    private final int contextDim;
    private final int numBands;
    private final BanditFeatureMapMode mode;

    public BanditFeatureEncoder(int contextDim, int numBands, BanditFeatureMapMode mode) {
        if (contextDim <= 0) {
            throw new IllegalArgumentException("contextDim must be positive");
        }
        if (numBands <= 0) {
            throw new IllegalArgumentException("numBands must be positive");
        }
        if (mode == null) {
            throw new IllegalArgumentException("mode must not be null");
        }

        this.contextDim = contextDim;
        this.numBands = numBands;
        this.mode = mode;
    }

    public int getEncodedDimension() {
        int additiveDim = contextDim + numBands;
        if (mode == BanditFeatureMapMode.FULL_INTERACTION) {
            return additiveDim + (contextDim * numBands);
        }
        return additiveDim;
    }

    public double[] encode(double[] context, int band) {
        validateContext(context);
        validateBand(band);

        double[] encoded = new double[getEncodedDimension()];
        System.arraycopy(context, 0, encoded, getContextOffset(), contextDim);
        encoded[getBandBiasOffset() + band] = 1.0;

        if (hasInteractionBlock()) {
            System.arraycopy(context, 0, encoded, getInteractionOffset() + (band * contextDim), contextDim);
        }

        return encoded;
    }

    public double[] buildPriorVariances(double contextPriorVariance,
            double bandBiasPriorVariance,
            double interactionPriorVariance) {
        if (contextPriorVariance <= 0.0 || bandBiasPriorVariance <= 0.0 || interactionPriorVariance <= 0.0) {
            throw new IllegalArgumentException("All prior variances must be positive");
        }

        double[] priors = new double[getEncodedDimension()];

        for (int i = 0; i < contextDim; i++) {
            priors[getContextOffset() + i] = contextPriorVariance;
        }
        for (int i = 0; i < numBands; i++) {
            priors[getBandBiasOffset() + i] = bandBiasPriorVariance;
        }

        if (hasInteractionBlock()) {
            for (int i = getInteractionOffset(); i < priors.length; i++) {
                priors[i] = interactionPriorVariance;
            }
        }

        return priors;
    }

    public BanditFeatureMapMode getMode() {
        return mode;
    }

    int getContextDim() {
        return contextDim;
    }

    int getNumBands() {
        return numBands;
    }

    int getContextOffset() {
        return 0;
    }

    int getBandBiasOffset() {
        return contextDim;
    }

    int getInteractionOffset() {
        return contextDim + numBands;
    }

    boolean hasInteractionBlock() {
        return mode == BanditFeatureMapMode.FULL_INTERACTION;
    }

    private void validateContext(double[] context) {
        if (context == null) {
            throw new IllegalArgumentException("context must not be null");
        }
        if (context.length != contextDim) {
            throw new IllegalArgumentException("Context dimension mismatch");
        }
    }

    private void validateBand(int band) {
        if (band < 0 || band >= numBands) {
            throw new IllegalArgumentException("Band index out of range");
        }
    }
}
