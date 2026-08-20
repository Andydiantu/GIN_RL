package gin.util.rl_mutation_choose;

import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import org.pmw.tinylog.Logger;

import java.util.List;

public class ContextVectorExtraction {
    public static final int METHOD_CONTEXT_DIM = 6;
    private static final String[] FEATURE_NAMES = {
            "Control",
            "Calc",
            "State",
            "Call",
            "Literal",
            "Structure"
    };

    private double[] featureMeans;
    private double[] featureStdDevs;

    public ContextVectorExtraction() {
    }

    /**
     * Extracts a general-purpose feature vector based on "Syntactic Category
     * Density".
     * Returns 6 features representing the density of different code actions.
     * All features sum to approximately 1.0 (excluding rounding/node overlaps).
     *
     * 1. Control Density (Logic)
     * 2. Calc Density (Math)
     * 3. State Density (Variables/Assignments)
     * 4. Call Density (Delegation)
     * 5. Literal Density (Data/Configuration)
     * 6. Structure Density (Error handling/Concurrency)
     */
    public double[] getRawFeatures(Node rootNode, int stepCount) {
        return getMethodDensityFeatures(rootNode);
    }

    private double[] getMethodDensityFeatures(Node rootNode) {
        List<Node> allNodes = rootNode.findAll(Node.class);
        double totalNodes = (double) allNodes.size();

        if (totalNodes == 0) {
            return new double[] { 0, 0, 0, 0, 0, 0 };
        }

        // 1. Control Density: Logic Flow
        long controlCount = allNodes.stream()
                .filter(n -> n instanceof IfStmt || n instanceof SwitchStmt || n instanceof SwitchEntry ||
                        n instanceof ForStmt || n instanceof WhileStmt || n instanceof DoStmt ||
                        n instanceof BreakStmt || n instanceof ContinueStmt || n instanceof ReturnStmt)
                .count();

        // 2. Calc Density: Math & Transformations
        long calcCount = allNodes.stream().filter(n -> {
            if (n instanceof BinaryExpr) {
                BinaryExpr.Operator op = ((BinaryExpr) n).getOperator();
                return op == BinaryExpr.Operator.PLUS || op == BinaryExpr.Operator.MINUS ||
                        op == BinaryExpr.Operator.MULTIPLY || op == BinaryExpr.Operator.DIVIDE ||
                        op == BinaryExpr.Operator.REMAINDER ||
                        op == BinaryExpr.Operator.BINARY_AND || op == BinaryExpr.Operator.BINARY_OR ||
                        op == BinaryExpr.Operator.XOR || op == BinaryExpr.Operator.LEFT_SHIFT ||
                        op == BinaryExpr.Operator.SIGNED_RIGHT_SHIFT || op == BinaryExpr.Operator.UNSIGNED_RIGHT_SHIFT;
            }
            return n instanceof UnaryExpr;
        }).count();

        // 3. State Density: Assignments & Variables
        long stateCount = allNodes.stream().filter(n -> n instanceof AssignExpr || n instanceof VariableDeclarator)
                .count();

        // 4. Call Density: Method Calls & Object Creation
        long callCount = allNodes.stream().filter(n -> n instanceof MethodCallExpr || n instanceof ObjectCreationExpr)
                .count();

        // 5. Literal Density: Hardcoded Data
        long literalCount = allNodes.stream().filter(n -> n instanceof LiteralExpr // Covers String, Integer, Boolean,
                                                                                   // Null, etc.
        ).count();

        // 6. Structure Density: Error Handling & Concurrency (Fragile Code)
        long structureCount = allNodes.stream()
                .filter(n -> n instanceof TryStmt || n instanceof CatchClause || n instanceof ThrowStmt ||
                        n instanceof SynchronizedStmt)
                .count();

        // Calculate Densities
        return new double[] {
                controlCount / totalNodes,
                calcCount / totalNodes,
                stateCount / totalNodes,
                callCount / totalNodes,
                literalCount / totalNodes,
                structureCount / totalNodes
        };
    }

    public void calculateFeatureStatistics(List<double[]> allFeatures) {
        if (allFeatures.isEmpty()) {
            return;
        }

        int numFeatures = allFeatures.get(0).length;
        featureMeans = new double[numFeatures];
        featureStdDevs = new double[numFeatures];

        // Calculate means
        for (int i = 0; i < numFeatures; i++) {
            double sum = 0;
            for (double[] features : allFeatures) {
                sum += features[i];
            }
            featureMeans[i] = sum / allFeatures.size();
        }

        // Calculate standard deviations
        for (int i = 0; i < numFeatures; i++) {
            double sumSqDiff = 0;
            for (double[] features : allFeatures) {
                double diff = features[i] - featureMeans[i];
                sumSqDiff += diff * diff;
            }
            featureStdDevs[i] = Math.sqrt(sumSqDiff / allFeatures.size());

            // Avoid division by zero
            if (featureStdDevs[i] == 0) {
                featureStdDevs[i] = 1.0;
            }
        }

        Logger.info("Context Feature Statistics (Syntactic Densities):");
        for (int i = 0; i < numFeatures; i++) {
            String name = (i < FEATURE_NAMES.length) ? FEATURE_NAMES[i] : "Feature " + i;
            Logger.info(String.format("%s: mean = %.4f, std dev = %.4f", name, featureMeans[i], featureStdDevs[i]));
        }
    }

    public double[] getNormalisedContextVector(Node rootNode, int stepCount) {
        return normaliseFeatures(getRawFeatures(rootNode, stepCount));
    }

    private double[] normaliseFeatures(double[] rawFeatures) {

        if (featureMeans == null || featureStdDevs == null) {
            Logger.warn("Feature means/std devs not initialized used, returning raw features.");
            return rawFeatures;
        }

        if (featureMeans.length != rawFeatures.length || featureStdDevs.length != rawFeatures.length) {
            Logger.warn("Feature dimension mismatch during normalization, returning raw features.");
            return rawFeatures;
        }

        double[] normalizedFeatures = new double[rawFeatures.length];
        for (int i = 0; i < rawFeatures.length; i++) {
            normalizedFeatures[i] = (rawFeatures[i] - featureMeans[i]) / featureStdDevs[i];
        }

        return normalizedFeatures;
    }

    // Overloaded method that includes recursive detection
    public double[] getNormalisedContextVector(Node rootNode, int stepCount, String methodName) {
        return getNormalisedContextVector(rootNode, stepCount);
    }

    public double[] getDummyContextVector() {
        return getDummyContextVector(METHOD_CONTEXT_DIM);
    }

    public double[] getDummyContextVector(int contextDim) {
        if (contextDim <= 0) {
            throw new IllegalArgumentException("contextDim must be positive");
        }

        double[] dummyContext = new double[contextDim];
        dummyContext[0] = 1.0;
        return dummyContext;
    }

}
