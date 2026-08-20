package gin.util.rl_mutation_choose;

import gin.SourceFileTree;
import gin.test.UnitTestResultSet;
import org.pmw.tinylog.Logger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class RlMutiAgentSystem {
    static final int WARM_UP_HORIZON = 500;
    static final double DIVERSITY_WEIGHT = 1.0;
    static final double FITNESS_WEIGHT = 1.0;

    private final MutiBanded mutiBanded;
    private final DiversityAgent diversityAgent;
    private final FitnessAgent fitnessAgent;
    private final BanditFeatureMapMode featureMapMode;
    private final int contextDim;
    private final int numBands;
    private final Set<String> uniquePatches;
    private final ContextVectorExtraction contextVectorExtraction;
    private final HashMap<Integer, Integer> mutationCount;

    private double[] currentContextVector;

    public RlMutiAgentSystem(int numBands, int numFeatures, int topK) {
        this(numBands, numFeatures, topK, BanditFeatureMapMode.ADDITIVE);
    }

    public RlMutiAgentSystem(int numBands, int numFeatures, int topK, BanditFeatureMapMode featureMapMode) {
        this.mutiBanded = new MutiBanded(numBands, numFeatures, topK, featureMapMode);
        this.diversityAgent = new DiversityAgent(numBands, numFeatures, featureMapMode);
        this.fitnessAgent = new FitnessAgent(numBands, numFeatures, featureMapMode);
        this.featureMapMode = featureMapMode;
        this.contextDim = numFeatures;
        this.numBands = numBands;
        this.uniquePatches = new HashSet<>();
        this.contextVectorExtraction = new ContextVectorExtraction();
        this.mutationCount = new HashMap<>();
        for (int i = 0; i < numBands; i++) {
            this.mutationCount.put(i, 0);
        }
    }

    public double[] update(int band, UnitTestResultSet results, double fitnessImprovement) {
        if (!Double.isFinite(fitnessImprovement) || fitnessImprovement < 0.0) {
            throw new IllegalArgumentException("fitnessImprovement must be finite and non-negative");
        }

        int validityLabel = isTestPassing(results) ? 1 : 0;
        mutiBanded.update(band, currentContextVector, validityLabel, mutationCount);

        int diversityLabel = 0;
        if (validityLabel == 1) {
            diversityLabel = uniquePatches.add(results.getPatch().toString()) ? 1 : 0;
            diversityAgent.update(band, currentContextVector, diversityLabel, mutationCount);
            fitnessAgent.update(band, currentContextVector, fitnessImprovement);
        }

        return new double[] {
                mutiBanded.predictProbMean(band, currentContextVector),
                diversityLabel,
                validityLabel == 1 ? fitnessImprovement : 0.0
        };
    }

    public double[] update(int band, UnitTestResultSet results) {
        return update(band, results, 0.0);
    }

    public int selectBand(SourceFileTree sourceFileTree, int patchNumber) {
        double[] extractedContext = this.contextVectorExtraction
                .getNormalisedContextVector(sourceFileTree.getTargetMethodRootNode().get(0), patchNumber);
        this.currentContextVector = getBanditContextVector(extractedContext);
        return selectBandForCurrentContext(patchNumber);
    }

    public int selectBandLocalSearch(SourceFileTree sourceFileTree, int patchNumber) {
        double[] extractedContext = this.contextVectorExtraction
                .getNormalisedContextVector(sourceFileTree.getTargetMethodRootNode().get(0), patchNumber);
        this.currentContextVector = getBanditContextVector(extractedContext);
        return selectBandForCurrentContext(patchNumber);
    }

    private int selectBandForCurrentContext(int searchStep) {
        double[] sampledValidityScore = this.mutiBanded.selectBand(this.currentContextVector, this.mutationCount);
        Logger.info("Sampled validity probabilities: " + Arrays.toString(sampledValidityScore));

        int[] allBands = new int[numBands];
        for (int i = 0; i < numBands; i++) {
            allBands[i] = i;
        }

        double[] sampledDiversityProb = this.diversityAgent.selectBand(
                this.currentContextVector, allBands, this.mutationCount);
        double[] sampledFitnessImprovement = this.fitnessAgent.selectBand(this.currentContextVector, allBands);

        Logger.info("Sampled diversity probabilities: " + Arrays.toString(sampledDiversityProb));
        Logger.info("Sampled fitness improvements: " + Arrays.toString(sampledFitnessImprovement));

        double[] combinedScore = new double[sampledValidityScore.length];
        double lambda = warmUpCoefficient(searchStep);
        for (int i = 0; i < combinedScore.length; i++) {
            combinedScore[i] = combineScore(
                    sampledValidityScore[i],
                    sampledDiversityProb[i],
                    sampledFitnessImprovement[i],
                    lambda);
        }
        Logger.info("Warm-up coefficient: " + lambda);
        Logger.info("Combined scores: " + Arrays.toString(combinedScore));

        int maxIndex = 0;
        double maxScore = combinedScore[0];
        for (int k = 1; k < combinedScore.length; k++) {
            if (combinedScore[k] > maxScore) {
                maxIndex = k;
                maxScore = combinedScore[k];
            }
        }
        Logger.info("Selected band: " + maxIndex);
        this.mutationCount.put(maxIndex, this.mutationCount.get(maxIndex) + 1);
        return maxIndex;
    }

    public int selectBandRandomSearch(SourceFileTree sourceFileTree, int patchNumber) {
        return selectBand(sourceFileTree, patchNumber);
    }

    public double[] getRawFeatures(SourceFileTree sourceFileTree, int patchNumber) {
        return this.contextVectorExtraction.getRawFeatures(sourceFileTree.getTargetMethodRootNode().get(0),
                patchNumber);
    }

    public void initializeFeatureStatistics(List<double[]> allFeatures) {
        this.contextVectorExtraction.calculateFeatureStatistics(allFeatures);
    }

    private double[] getBanditContextVector(double[] extractedContext) {
        if (featureMapMode == BanditFeatureMapMode.UNCONTEXTUAL) {
            return contextVectorExtraction.getDummyContextVector(contextDim);
        }
        return extractedContext;
    }

    static double warmUpCoefficient(int searchStep) {
        return Math.min(1.0, Math.max(0, searchStep) / (double) WARM_UP_HORIZON);
    }

    static double combineScore(double validityProb,
            double diversityProb,
            double fitnessImprovement,
            double lambda) {
        double fullScore = validityProb *
                (DIVERSITY_WEIGHT * diversityProb + FITNESS_WEIGHT * fitnessImprovement);
        return (1.0 - lambda) * validityProb + lambda * fullScore;
    }

    private static boolean isTestPassing(UnitTestResultSet results) {
        return results.getCleanCompile() && results.allTestsSuccessful();
    }

    public void logBanditWeights() {
        this.mutiBanded.logFeatureWeights();
    }

    public BanditFeatureMapMode getFeatureMapMode() {
        return featureMapMode;
    }

    public int getEncodedFeatureDimension() {
        return mutiBanded.getEncodedFeatureDimension();
    }
}
