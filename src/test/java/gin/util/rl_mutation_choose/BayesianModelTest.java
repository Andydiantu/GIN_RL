package gin.util.rl_mutation_choose;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class BayesianModelTest {

    @Test
    public void sampledLogisticCoefficientsCanBeReusedAcrossOperators() {
        BayesianLogisticRegression model = new BayesianLogisticRegression(3, 1.0);
        double[] coefficients = model.sampleCoefficients();

        assertEquals(3, coefficients.length);
        double expected = 1.0 / (1.0 + Math.exp(-coefficients[0]));
        assertEquals(expected,
                model.predictProb(new double[] { 1.0, 0.0, 0.0 }, coefficients), 1e-12);
    }

    @Test
    public void sampledLinearCoefficientsCanBeReusedAcrossOperators() {
        BayesianLinearRegression model = new BayesianLinearRegression(3, 5.0, 5.0);
        model.update(new double[] { 1.0, 0.0, 1.0 }, 0.25);
        double[] coefficients = model.sampleCoefficients();

        assertEquals(3, coefficients.length);
        assertEquals(coefficients[0] + coefficients[2],
                model.predict(new double[] { 1.0, 0.0, 1.0 }, coefficients), 1e-12);
    }

    @Test(expected = IllegalArgumentException.class)
    public void fitnessModelRejectsNegativeImprovement() {
        FitnessAgent model = new FitnessAgent(2, 6);
        model.update(0, new double[6], -0.1);
    }
}
