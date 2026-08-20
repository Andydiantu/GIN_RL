package gin.util.rl_mutation_choose;

public interface LogisticRegression {

    void update(double[] xArray, int y);

    double predictProbSample(double[] xArray);

    default double predictProbMean(double[] xArray) {
        return predictProbSample(xArray);
    }
}
