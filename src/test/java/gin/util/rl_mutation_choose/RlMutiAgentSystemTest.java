package gin.util.rl_mutation_choose;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class RlMutiAgentSystemTest {

    @Test
    public void warmUpCoefficientFollowsPaperSchedule() {
        assertEquals(0.0, RlMutiAgentSystem.warmUpCoefficient(0), 0.0);
        assertEquals(0.5, RlMutiAgentSystem.warmUpCoefficient(250), 0.0);
        assertEquals(1.0, RlMutiAgentSystem.warmUpCoefficient(500), 0.0);
        assertEquals(1.0, RlMutiAgentSystem.warmUpCoefficient(1000), 0.0);
    }

    @Test
    public void combinedScoreInterpolatesValidityAndFullFactorisedScore() {
        double validity = 0.8;
        double diversity = 0.25;
        double fitness = 0.5;
        double fullScore = validity * (diversity + fitness);

        assertEquals(validity,
                RlMutiAgentSystem.combineScore(validity, diversity, fitness, 0.0), 1e-12);
        assertEquals((validity + fullScore) / 2.0,
                RlMutiAgentSystem.combineScore(validity, diversity, fitness, 0.5), 1e-12);
        assertEquals(fullScore,
                RlMutiAgentSystem.combineScore(validity, diversity, fitness, 1.0), 1e-12);
    }
}
