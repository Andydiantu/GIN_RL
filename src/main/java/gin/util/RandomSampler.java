package gin.util;

import com.sampullara.cli.Args;
import com.sampullara.cli.Argument;
import gin.Patch;
import gin.SourceFile;
import gin.SourceFileTree;
import gin.edit.Edit;
import gin.edit.Edit.EditType;
import gin.edit.llm.LLMMaskedStatement;
import gin.edit.llm.LLMReplaceStatement;
import gin.edit.llm.LLMTargetSelector;
import gin.edit.statement.CopyStatement;
import gin.edit.statement.DeleteStatement;
import gin.test.UnitTestResultSet;
import gin.util.rl_mutation_choose.BanditFeatureMapMode;
import gin.util.rl_mutation_choose.RlMutiAgentSystem;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.Node;

import org.apache.commons.rng.simple.JDKRandomBridge;
import org.apache.commons.rng.simple.RandomSource;
import org.pmw.tinylog.Logger;

import java.io.File;
import java.io.Serial;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

/**
 * Random sampler.
 * <p>
 * Creates patchNumber random method patches of size 1:patchSize
 */

public class RandomSampler extends Sampler {

    @Serial
    private static final long serialVersionUID = 5754760811598365140L;

    @Argument(alias = "et", description = "Edit type: this can be a member of the EditType enum (LINE,STATEMENT,MATCHED_STATEMENT,MODIFY_STATEMENT); the fully qualified name of a class that extends gin.edit.Edit, or a comma separated list of both")
    protected String editType = EditType.LINE.toString();

    @Argument(alias = "ps", description = "Number of edits per patch")
    protected Integer patchSize = 1;

    @Argument(alias = "pn", description = "Number of patches")
    protected Integer patchNumber = 10;

    @Argument(alias = "rm", description = "Random seed for method selection")
    protected Integer methodSeed = 123;

    @Argument(alias = "rp", description = "Random seed for edit type selection")
    protected Integer patchSeed = 123;

    @Argument(alias = "pb", description = "Probablity of combiend")
    protected Double combinedProbablity = 0.5;

    @Argument(alias = "rl", description = "if doing reinforcement learning based mutation selection")
    protected Boolean rl = false;

    @Argument(alias = "ds", description = "if doing LLM based destination selection")
    protected Boolean llmDestinationSelection = false;

    @Argument(alias = "fm", description = "RL bandit feature-map mode: ADDITIVE, FULL_INTERACTION, or UNCONTEXTUAL")
    protected BanditFeatureMapMode rlFeatureMapMode = BanditFeatureMapMode.ADDITIVE;

    // Whether to use LLM edits
    private boolean ifLLM = false;

    private Class<? extends Edit> LLMedit = null;

    private List<Class<? extends Edit>> NoneLLMedit = new ArrayList<>();

    private Random mutationRng;

    private RlMutiAgentSystem rlMutiAgentSystem;

    private LLMTargetSelector llmTargetSelector;

    /**
     * allowed edit types for sampling: parsed from editType
     */
    protected List<Class<? extends Edit>> editTypes;

    public RandomSampler(String[] args) {
        super(args);
        Args.parseOrExit(this, args);
        editTypes = Edit.parseEditClassesFromString(editType);
        Setup();
        printAdditionalArguments();
    }

    private void Setup() {

        mutationRng = new JDKRandomBridge(RandomSource.MT, Long.valueOf(patchSeed));

        if (editTypes.contains(LLMMaskedStatement.class) || editTypes.contains(LLMReplaceStatement.class)) {
            ifLLM = true;
            if (editTypes.contains(LLMMaskedStatement.class)) {
                LLMedit = LLMMaskedStatement.class;
            } else if (editTypes.contains(LLMReplaceStatement.class)) {
                LLMedit = LLMReplaceStatement.class;
            }

            for (Class<? extends Edit> edit : editTypes) {
                if (edit != LLMedit) {
                    NoneLLMedit.add(edit);
                }
            }
        }

        if (rl) {
            rlMutiAgentSystem = new RlMutiAgentSystem(editTypes.size(), 6, 3, rlFeatureMapMode);

            Logger.info("Calculating feature statistics for Z-score normalization...");
            List<double[]> allFeatures = new ArrayList<>();
            for (TargetMethod method : methodData) {
                try {
                    SourceFile sourceFile = SourceFile.makeSourceFileForEditTypes(editTypes,
                            method.getFileSource().getPath(),
                            Collections.singletonList(method.getMethodName()));

                    if (sourceFile instanceof SourceFileTree) {
                        allFeatures.add(rlMutiAgentSystem.getRawFeatures((SourceFileTree) sourceFile, 0));
                    }
                } catch (Exception e) {
                    Logger.warn("Failed to extract features for method " + method.toString() + ": " + e.getMessage());
                }
            }
            rlMutiAgentSystem.initializeFeatureStatistics(allFeatures);
            Logger.info("Feature statistics calculated from " + allFeatures.size() + " methods.");
        }

        if (llmDestinationSelection) {
            llmTargetSelector = new LLMTargetSelector();
        }

        Logger.info("=== RandomSampler ===");
        Logger.info("LLM edits: " + ifLLM);
        Logger.info("None LLM edits: " + NoneLLMedit.toString());
        Logger.info("LLM edit: " + LLMedit);
        Logger.info("=====================================");
    }

    // Constructor used for testing
    public RandomSampler(File projectDir, File methodFile) {
        super(projectDir, methodFile);
        editTypes = Edit.parseEditClassesFromString(editType);
        Setup();
    }

    public static void main(String[] args) {
        RandomSampler sampler = new RandomSampler(args);
        sampler.sampleMethods();
    }

    private void printAdditionalArguments() {
        Logger.info("Edit types: " + editTypes);
        Logger.info("Number of edits per patch: " + patchSize);
        Logger.info("Number of patches: " + patchNumber);
        Logger.info("Random seed for method selection: " + methodSeed);
        Logger.info("Random seed for edit type selection: " + patchSeed);
        Logger.info("Reinforcement learning: " + rl);
        Logger.info("LLM destination selection: " + llmDestinationSelection);
        Logger.info("RL feature map mode: " + rlFeatureMapMode);
    }

    /**
     * Generate a neighbouring patch, by either deleting an edit, or adding a new
     * one.
     *
     * @param patch Generate a neighbour of this patch.
     * @return A neighbouring patch.
     */
    Patch neighbour(Patch patch) {

        Patch neighbour = patch.clone();

        // if(ifLLM && NoneLLMedit.size() > 0){
        // if (mutationRng.nextFloat() > combinedProbablity) {

        // neighbour.addRandomEditOfClasses(mutationRng, Arrays.asList(LLMedit));
        // }
        // else {
        // neighbour.addRandomEditOfClasses(mutationRng, NoneLLMedit);
        // }
        // } else {
        neighbour.addRandomEditOfClasses(mutationRng, editTypes);
        // }

        return neighbour;
    }

    Patch neighbour(Patch patch, int band) {
        Patch neighbour = patch.clone();
        neighbour.addRandomEditOfClasses(mutationRng, Arrays.asList(editTypes.get(band)));
        return neighbour;
    }

    Patch neighbour(Patch patch, int band, SourceFileTree sourceFileTree) {
        Patch neighbour = patch.clone();

        // if band is -1, select a random band
        if (band == -1) {
            band = mutationRng.nextInt(editTypes.size());
        }

        Class<? extends Edit> editClass = editTypes.get(band);
        Logger.info("Edit class: " + editClass);

        // Destination selection is defined only for the traditional operators.
        // The LLM operators retain their own target-selection behaviour.
        if (editClass == LLMMaskedStatement.class || editClass == LLMReplaceStatement.class) {
            neighbour.addRandomEditOfClasses(mutationRng, Arrays.asList(editClass));
            return neighbour;
        }

        Node targetMethodRootNode = sourceFileTree.getTargetMethodRootNode().get(0);

        try {
            if (editClass == DeleteStatement.class) {
                int targetStatementID = llmTargetSelector.selectTargetSingle(targetMethodRootNode, editClass);
                Edit edit = editClass.getDeclaredConstructor(String.class, int.class)
                        .newInstance(sourceFileTree.getRelativePathToWorkingDir(), targetStatementID);
                neighbour.add(edit);
            } else if (editClass == CopyStatement.class) {
                int[] targetStatementIDs = llmTargetSelector.selectTargetMultiple(targetMethodRootNode, editClass);
                Edit edit = editClass
                        .getDeclaredConstructor(String.class, int.class, String.class, int.class, int.class)
                        .newInstance(sourceFileTree.getRelativePathToWorkingDir(), targetStatementIDs[0],
                                sourceFileTree.getRelativePathToWorkingDir(), targetStatementIDs[1],
                                targetStatementIDs[2]);
                neighbour.add(edit);
            } else {
                int[] targetStatementIDs = llmTargetSelector.selectTargetMultiple(targetMethodRootNode, editClass);
                Edit edit = editClass.getDeclaredConstructor(String.class, int.class, String.class, int.class)
                        .newInstance(sourceFileTree.getRelativePathToWorkingDir(), targetStatementIDs[0],
                                sourceFileTree.getRelativePathToWorkingDir(), targetStatementIDs[1]);
                neighbour.add(edit);
            }
        } catch (Exception e) {
            Logger.error("Error creating edit of type " + editClass.getSimpleName() + ": " + e.getMessage());
            // Fallback to random selection
            neighbour.addRandomEditOfClasses(mutationRng, Arrays.asList(editClass));
        }
        // Logger.info("Target statement ID: " + targetStatementID);

        return neighbour;
    }

    // Test function to printout statement IDs and nodes, currently not in use
    public Map<Integer, Statement> getStatementIDsAndNodes(Node node) {
        Map<Integer, Statement> statementMap = new HashMap<>();

        List<Statement> statements = node.findAll(Statement.class);

        for (Statement stmt : statements) {
            Integer id = stmt.containsData(SourceFileTree.NODEKEY_ID) ? stmt.getData(SourceFileTree.NODEKEY_ID)
                    : SourceFileTree.NODE_NULL_ID;
            statementMap.put(id, stmt);
        }

        // Print all statement-ID pairs
        Logger.info("=== Statement ID and Node Pairs ===");
        for (Map.Entry<Integer, Statement> entry : statementMap.entrySet()) {
            Integer id = entry.getKey();
            Statement stmt = entry.getValue();
            Logger.info("ID: " + id + " -> Statement: " + stmt.toString());
        }
        Logger.info("Total statements: " + statementMap.size());
        Logger.info("=====================================");

        return statementMap;
    }

    protected void sampleMethodsHook() {

        Random mrng = new JDKRandomBridge(RandomSource.MT, Long.valueOf(methodSeed));

        if (patchSize > 0) {

            writeHeader();

            int size = methodData.size();
            Map<Integer, Double> originalExecutionTimes = new HashMap<>();

            Logger.info("Start applying and testing random patches..");

            Logger.info("Number of patch: " + patchNumber);

            for (int i = 0; i < patchNumber; i++) {
                Random prng = new JDKRandomBridge(RandomSource.MT, patchSeed + (100000L * i));

                // Pick a random method
                TargetMethod method = methodData.get(mrng.nextInt(size));
                Integer methodID = method.getMethodID();
                File source = method.getFileSource();

                // Setup SourceFile for patching
                SourceFile sourceFile = SourceFile.makeSourceFileForEditTypes(editTypes, source.getPath(),
                        Collections.singletonList(method.getMethodName()));

                Patch patch = new Patch(sourceFile);
                int bandSelected = 0;

                if (rl) {
                    if (sourceFile instanceof SourceFileTree) {
                        SourceFileTree sourceFileTree = (SourceFileTree) sourceFile;
                        bandSelected = rlMutiAgentSystem.selectBandRandomSearch(sourceFileTree, i + 1);
                        Logger.info("Band selected: " + editTypes.get(bandSelected));
                        for (int j = 0; j < patchSize; j++) {
                            if (llmDestinationSelection) {
                                patch = neighbour(patch, bandSelected, sourceFileTree);
                            } else {
                                patch = neighbour(patch, bandSelected);
                            }
                        }
                    } else {
                        for (int j = 0; j < patchSize; j++) {
                            patch = neighbour(patch);
                        }
                    }
                } else {
                    for (int j = 0; j < patchSize; j++) {
                        if (llmDestinationSelection) {
                            SourceFileTree sourceFileTree = (SourceFileTree) sourceFile;
                            patch = neighbour(patch, -1, sourceFileTree);
                        } else {
                            patch = neighbour(patch);
                        }
                    }
                }

                Logger.info("Testing random patch " + patch + " for method: " + method + " with ID " + methodID);

                if (rl && !originalExecutionTimes.containsKey(methodID)) {
                    UnitTestResultSet originalResults = testPatchWithoutCounting(
                            method.getClassName(), method.getGinTests(), new Patch(sourceFile), null);
                    double originalExecutionTime = originalResults.getCleanCompile()
                            && originalResults.allTestsSuccessful()
                                    ? originalResults.totalExecutionTime()
                                    : Double.POSITIVE_INFINITY;
                    originalExecutionTimes.put(methodID, originalExecutionTime);
                }

                // Test the patched source file
                UnitTestResultSet results = testPatch(method.getClassName(), method.getGinTests(), patch, null);

                if (rl) {
                    double normalizedFitnessImprovement = normaliseRuntimeImprovement(
                            originalExecutionTimes.get(methodID), results);
                    double[] patchReward = rlMutiAgentSystem.update(
                            bandSelected, results, normalizedFitnessImprovement);
                    Logger.info("Validity probability is " + patchReward[0]);
                    Logger.info("Diversity reward is " + patchReward[1]);
                    Logger.info("Normalized fitness improvement is " + patchReward[2]);
                    writeResults(results, methodID, patchReward[0], bandSelected);
                } else {
                    writeResults(results, methodID, 0, 0);
                }
            }
            if (rl) {
                rlMutiAgentSystem.logBanditWeights();
            }
            Logger.info("Results saved to: " + outputFile);

        } else {
            Logger.info("Number of edits  must be greater than 0.");
        }
    }

    RlMutiAgentSystem getRlMutiAgentSystem() {
        return rlMutiAgentSystem;
    }

    private double normaliseRuntimeImprovement(double originalExecutionTime, UnitTestResultSet results) {
        if (!Double.isFinite(originalExecutionTime) || originalExecutionTime <= 0.0
                || !results.getCleanCompile() || !results.allTestsSuccessful()) {
            return 0.0;
        }
        double improvement = originalExecutionTime - results.totalExecutionTime();
        return Math.max(0.0, improvement / originalExecutionTime);
    }

}
