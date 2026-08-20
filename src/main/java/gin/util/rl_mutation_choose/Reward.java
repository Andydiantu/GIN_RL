package gin.util.rl_mutation_choose;

import gin.test.UnitTestResultSet;

public class Reward {
    private final double wallTimeWeight;

    // Currently the wallTimeWeight is not used
    public Reward(double wallTimeWeight) {
        this.wallTimeWeight = wallTimeWeight;
    }

    public double getWallTimeWeight() {
        return wallTimeWeight;
    }

    public double calculateReward(UnitTestResultSet results) {
        double reward = 0;
        if (!results.getValidPatch()) {
            reward = -0.2;
        } else if (!results.getCleanCompile()) {
            reward = -0.1;
        } else {
            reward = results.testPassRate();
            if (results.testPassRate() == 1) {
                reward += 4; // extra reward for passing all tests
            }
        }
        return reward;
    }
}