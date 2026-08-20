package gin.util.rl_mutation_choose;

public interface LinearRegression {
    void update(double[] xArray, double y);

    double predictSample(double[] xArray);
}