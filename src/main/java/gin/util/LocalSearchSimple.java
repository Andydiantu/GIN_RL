package gin.util;

import gin.Patch;
import gin.SourceFile;
import gin.SourceFileTree;
import gin.edit.Edit;
import gin.edit.line.CopyLine;
import gin.edit.line.DeleteLine;
import gin.edit.line.LineEdit;
import gin.edit.llm.LLMMaskedStatement;
import gin.edit.llm.LLMReplaceStatement;
import gin.edit.statement.CopyStatement;
import gin.edit.statement.DeleteStatement;
import gin.edit.llm.LLMTargetSelector;
import gin.test.UnitTest;
import gin.test.UnitTestResultSet;
import gin.util.rl_mutation_choose.ContextVectorExtraction;
import gin.util.rl_mutation_choose.RlMutiAgentSystem;
import gin.SourceFileTree;
import com.github.javaparser.ast.Node;


import org.pmw.tinylog.Logger;

import com.fasterxml.jackson.annotation.JsonTypeInfo.None;
import com.sampullara.cli.Argument;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.rng.simple.JDKRandomBridge;
import org.apache.commons.rng.simple.RandomSource;

/**
 * Method-based LocalSearchSimple search.
 */

public abstract class LocalSearchSimple extends GP {

    // Percentage of population size to be selected during tournament selection
    private static final double TOURNAMENT_PERCENTAGE = 0.2;
    // Probability of adding an edit during uniform crossover
    private static final double MUTATE_PROBABILITY = 0.5;
    private static final int LOCAL_SEARCH_TOP_K = 3;
    private static final String FITNESS_COLUMN = "Fitness";
    private static final String FITNESS_IMPROVEMENT_COLUMN = "FitnessImprovement";

    // Whether to use LLM edits
    private boolean ifLLM = false;

    private Class <? extends Edit> LLMedit = null;

    private List<Class <? extends Edit>> NoneLLMedit = new ArrayList<>();

    private RlMutiAgentSystem rlMutiAgentSystem;

    private LLMTargetSelector llmTargetSelector;

    public LocalSearchSimple(String[] args) {
        super(args);
        SetLLMedits();
        initializeRlSystem();

        if (llmDestinationSelection) {
            llmTargetSelector = new LLMTargetSelector();
        }
    }

    // Constructor used for testing
    public LocalSearchSimple(File projectDir, File methodFile) {
        super(projectDir, methodFile); 
        SetLLMedits();
        initializeRlSystem();

        if (llmDestinationSelection) {
            llmTargetSelector = new LLMTargetSelector();
        }
    }

    private void initializeRlSystem() {
        if (!rl) {
            return;
        }

        rlMutiAgentSystem = new RlMutiAgentSystem(super.editTypes.size(),
                ContextVectorExtraction.METHOD_CONTEXT_DIM,
                LOCAL_SEARCH_TOP_K,
                rlFeatureMapMode);

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

    private void SetLLMedits () {
        if (super.editTypes.contains(LLMMaskedStatement.class) || super.editTypes.contains(LLMReplaceStatement.class)) {
            ifLLM = true;
            if (super.editTypes.contains(LLMMaskedStatement.class)) {
                LLMedit = LLMMaskedStatement.class;
            } else if (super.editTypes.contains(LLMReplaceStatement.class)) {
                LLMedit = LLMReplaceStatement.class;
            }

            for (Class <? extends Edit> edit : super.editTypes) {
                if (edit != LLMedit) {
                    NoneLLMedit.add(edit);
                }
            }
        }


        Logger.info("=== LocalSearchSimple ===");
        Logger.info("LLM edits: " + ifLLM);
        Logger.info("None LLM edits: " + NoneLLMedit.toString());
        Logger.info("LLM edit: " + LLMedit);
        Logger.info("=====================================");


    }

    // Whatever initialisation needs to be done for fitness calculations
    @Override
    protected abstract UnitTestResultSet initFitness(String className, List<UnitTest> tests, Patch origPatch);

    // Calculate fitness
    @Override
    protected abstract double fitness(UnitTestResultSet results);

    // Calculate fitness threshold, for selection to the next generation
    @Override
    protected abstract boolean fitnessThreshold(UnitTestResultSet results, double orig);

    @Override
    protected String[] outputHeader() {
        return appendColumns(super.outputHeader(), FITNESS_COLUMN, FITNESS_IMPROVEMENT_COLUMN);
    }

    private void writeLocalSearchResults(UnitTestResultSet results, Integer methodID, double fitness,
                                         double fitnessImprovement, int bandSelected) {
        writeResults(results,
                methodID,
                fitnessImprovement,
                bandSelected,
                Double.toString(fitness),
                Double.toString(fitnessImprovement));
    }

    /*============== Implementation of abstract methods  ==============*/

    /*====== Interleaved Search ======*/

    @Override
    protected void sampleMethodsHook() {

        if ((indNumber < 1) || (genNumber < 1)) {
            Logger.info("Please enter a positive number of generations and individuals.");
            return;
        }

        writeHeader();

        int numberToSearch = Math.min(methodData.size(), methodNumber);
        List<TargetMethod> methods = methodData.subList(0, numberToSearch);
        int perMethodMutationBudget = Math.max(0, indNumber - 1);

        Map<TargetMethod, Double> origFitness = new HashMap<>();
        Map<TargetMethod, Patch> bestPatches = new HashMap<>();
        Map<TargetMethod, Double> bestFitness = new HashMap<>();

        Logger.info("Initializing " + methods.size() + " methods for interleaved local search.");
        for (TargetMethod method : methods) {
            SourceFile sourceFile = SourceFile.makeSourceFileForEditTypes(editTypes,
                    method.getFileSource().getPath(),
                    Collections.singletonList(method.getMethodName()));
            Patch origPatch = new Patch(sourceFile);
            UnitTestResultSet results = initFitness(method.getClassName(), method.getGinTests(), origPatch);
            double orig = fitness(results);

            origFitness.put(method, orig);
            bestPatches.put(method, origPatch);
            bestFitness.put(method, orig);
            writeLocalSearchResults(results, method.getMethodID(), orig, 0.0, 0);
        }

        List<TargetMethod> searchSchedule = new ArrayList<>();
        for (TargetMethod method : methods) {
            for (int i = 0; i < perMethodMutationBudget; i++) {
                searchSchedule.add(method);
            }
        }
        Random scheduleRng = new JDKRandomBridge(RandomSource.MT, Long.valueOf(individualSeed));
        Collections.shuffle(searchSchedule, scheduleRng);

        Logger.info("Running " + searchSchedule.size() + " interleaved local-search evaluations.");
        for (int evaluation = 1; evaluation <= searchSchedule.size(); evaluation++) {
            TargetMethod method = searchSchedule.get(evaluation - 1);
            Patch currentBestPatch = bestPatches.get(method);
            double originalFitness = origFitness.get(method);
            double incumbentFitness = bestFitness.get(method);
            int bandSelected = 0;
            Patch patch;

            if (rl) {
                if (currentBestPatch.getSourceFile() instanceof SourceFileTree) {
                    SourceFileTree sourceFileTree = (SourceFileTree) currentBestPatch.getSourceFile();
                    bandSelected = rlMutiAgentSystem.selectBandLocalSearch(sourceFileTree, evaluation);
                    Logger.info("Band selected: " + super.editTypes.get(bandSelected));

                    if (llmDestinationSelection) {
                        patch = neighbour(currentBestPatch, bandSelected, sourceFileTree);
                    } else {
                        patch = neighbour(currentBestPatch, bandSelected);
                    }
                } else {
                    patch = neighbour(currentBestPatch);
                }
            } else {
                if (llmDestinationSelection) {
                    SourceFileTree sourceFileTree = (SourceFileTree) currentBestPatch.getSourceFile();
                    patch = neighbour(currentBestPatch, -1, sourceFileTree);
                } else {
                    patch = neighbour(currentBestPatch);
                }
            }

            UnitTestResultSet results = testPatch(method.getClassName(), method.getGinTests(), patch, null);
            double newFitness = fitness(results);
            double incumbentImprovement = compareFitness(newFitness, incumbentFitness);
            double fitnessImprovement = compareFitness(newFitness, originalFitness);
            double normalizedFitnessImprovement = normaliseImprovement(fitnessImprovement, originalFitness);

            if (rl) {
                double[] patchReward = rlMutiAgentSystem.update(
                        bandSelected, results, normalizedFitnessImprovement);

                Logger.info("Validity probability is " + patchReward[0]);
                Logger.info("Diversity reward is " + patchReward[1]);
                Logger.info("Normalized fitness improvement is " + patchReward[2]);
            }

            Logger.info("Method: " + method + ", new fitness: " + newFitness);
            Logger.info("Improvement: " + compareFitness(newFitness, originalFitness));
            writeLocalSearchResults(results, method.getMethodID(), newFitness, fitnessImprovement, bandSelected);

            if (incumbentImprovement > 0) {
                bestFitness.put(method, newFitness);
                bestPatches.put(method, patch);
            }
        }

        if (rl) {
            rlMutiAgentSystem.logBanditWeights();
        }
    }

    /*====== Search ======*/

    // Simple GP search (based on Simple)
    protected void search(TargetMethod method, Patch origPatch) {

        Logger.info("Runnning best-first local search.");

        String className = method.getClassName();
        List<UnitTest> tests = method.getGinTests();

        // Run original code
        UnitTestResultSet results = initFitness(className, tests, origPatch);

        // Calculate fitness and record result, including fitness improvement (currently 0)
        double orig = fitness(results);
        writeLocalSearchResults(results, method.getMethodID(), orig, 0.0, 0);

        // Keep best
        double best = orig;
        Patch bestPatch = origPatch;
        for (int i = 1; i < indNumber; i++) {
            int bandSelected = 0;

            Patch patch = null;

            if (rl) {

                if (bestPatch.getSourceFile() instanceof SourceFileTree) {
                    SourceFileTree sourceFileTree = (SourceFileTree) bestPatch.getSourceFile();
                    bandSelected = rlMutiAgentSystem.selectBandLocalSearch(sourceFileTree, i);
                    Logger.info("Band selected: " + super.editTypes.get(bandSelected));

                    if (llmDestinationSelection) {
                        patch = neighbour(bestPatch, bandSelected, sourceFileTree);
                    } else {
                        patch = neighbour(bestPatch, bandSelected);
                    }
                } else {
                    patch = neighbour(bestPatch);
                }
            } else {
                if (llmDestinationSelection) {
                    SourceFileTree sourceFileTree = (SourceFileTree) bestPatch.getSourceFile();
                    patch = neighbour(bestPatch, -1, sourceFileTree);
                } else {
                    patch = neighbour(bestPatch);
                }
            }

            // Calculate fitness
            results = testPatch(className, tests, patch, null);
            double newFitness = fitness(results);
            double incumbentImprovement = compareFitness(newFitness, best);
            double fitnessImprovement = compareFitness(newFitness, orig);
            double normalizedFitnessImprovement = normaliseImprovement(fitnessImprovement, orig);

            if (rl) {
                double[] patchReward = rlMutiAgentSystem.update(
                        bandSelected, results, normalizedFitnessImprovement);

                Logger.info("Validity probability is " + patchReward[0]);
                Logger.info("Diversity reward is " + patchReward[1]);
                Logger.info("Normalized fitness improvement is " + patchReward[2]);
            }

            Logger.info("New fitness: " + newFitness);
            Logger.info("Improvement: " + compareFitness(newFitness, orig));
            writeLocalSearchResults(results, method.getMethodID(), newFitness, fitnessImprovement, bandSelected);
            

            // Check if better
            if (incumbentImprovement > 0) {
                best = newFitness;
                bestPatch = patch;
            }
        }


        if (rl) {
            rlMutiAgentSystem.logBanditWeights();
        }
    }

    RlMutiAgentSystem getRlMutiAgentSystem() {
        return rlMutiAgentSystem;
    }

    private double normaliseImprovement(double improvement, double originalFitness) {
        if (improvement <= 0.0) {
            return 0.0;
        }

        double denominator = Math.max(1.0, originalFitness);
        return Math.min(1.0, improvement / denominator);
    }

    /*====== GP Operators ======*/

    /**
     * Generate a neighbouring patch, by either deleting an edit, or adding a new one.
     *
     * @param patch Generate a neighbour of this patch.
     * @return A neighbouring patch.
     */
    Patch neighbour(Patch patch) {

        Patch neighbour = patch.clone();

        // if(ifLLM && NoneLLMedit.size() > 0){
        //     Logger.info("LLM edit" + super.combinedProbablity);
        //     if (neighbour.size() > 0 && super.mutationRng.nextFloat() > super.combinedProbablity) {
        //         neighbour.addRandomEditOfClasses(super.mutationRng, Arrays.asList(LLMedit));
        //     } 
        //     else {
        //         neighbour.addRandomEditOfClasses(super.mutationRng, NoneLLMedit);
        //     }
        // } else {
            neighbour.addRandomEditOfClasses(super.mutationRng, super.editTypes);
        // }


        return neighbour;

    }

    Patch neighbour(Patch patch, int band) {
        Patch neighbour = patch.clone();
        neighbour.addRandomEditOfClasses(super.mutationRng, Arrays.asList(super.editTypes.get(band)));
        return neighbour;
    }

    Patch neighbour(Patch patch, int band, SourceFileTree sourceFileTree) {
        Patch neighbour = patch.clone();

        // if band is -1, select a random band
        if (band == -1) {
            band = super.mutationRng.nextInt(super.editTypes.size());
        }

        Class<? extends Edit> editClass = super.editTypes.get(band);
        Logger.info("Edit class: " + editClass);

        // Destination selection is defined only for the traditional operators.
        // The LLM operators retain their own target-selection behaviour.
        if (editClass == LLMMaskedStatement.class || editClass == LLMReplaceStatement.class) {
            neighbour.addRandomEditOfClasses(super.mutationRng, Arrays.asList(editClass));
            return neighbour;
        }

        Node targetMethodRootNode = sourceFileTree.getTargetMethodRootNode().get(0);

        try {
            if (editClass == DeleteStatement.class) {
                int targetStatementID = llmTargetSelector.selectTargetSingle(targetMethodRootNode, editClass);
                Edit edit = editClass.getDeclaredConstructor(String.class, int.class).newInstance(sourceFileTree.getRelativePathToWorkingDir(), targetStatementID);
                neighbour.add(edit);
            } else if (editClass == CopyStatement.class) {
                int[] targetStatementIDs = llmTargetSelector.selectTargetMultiple(targetMethodRootNode, editClass);
                Edit edit = editClass.getDeclaredConstructor(String.class, int.class, String.class, int.class, int.class).newInstance(sourceFileTree.getRelativePathToWorkingDir(), targetStatementIDs[0], sourceFileTree.getRelativePathToWorkingDir(), targetStatementIDs[1], targetStatementIDs[2]);
                neighbour.add(edit);
            } else {
                int[] targetStatementIDs = llmTargetSelector.selectTargetMultiple(targetMethodRootNode, editClass);
                Edit edit = editClass.getDeclaredConstructor(String.class, int.class, String.class, int.class).newInstance(sourceFileTree.getRelativePathToWorkingDir(), targetStatementIDs[0], sourceFileTree.getRelativePathToWorkingDir(), targetStatementIDs[1]);
                neighbour.add(edit);
            }
        } catch (Exception e) {
            Logger.error("Error creating edit of type " + editClass.getSimpleName() + ": " + e.getMessage());
            // Fallback to random selection
            neighbour.addRandomEditOfClasses(super.mutationRng, Arrays.asList(editClass));
        }
        // Logger.info("Target statement ID: " + targetStatementID);

        return neighbour;
    }


    // Adds a random edit of the given type with equal probability among allowed types
    // TODO: This is a bit of a hack, as it assumes that if only put one edit type, it is LLMReplaceStatement or LLMMaskedStatement
    protected Patch mutate(Patch oldPatch) {
        Patch patch = oldPatch.clone();
        patch.addRandomEditOfClasses(super.mutationRng, super.editTypes);

        return patch;
    }

    // Tournament selection for patches
    protected List<Patch> select(Map<Patch, Double> population, Patch origPatch, double origFitness) {

        List<Patch> patches = new ArrayList<>(population.keySet());
        if (patches.size() < super.indNumber) {
            population.put(origPatch, origFitness);
            while (patches.size() < super.indNumber) {
                patches.add(origPatch);
            }
        }
        List<Patch> selectedPatches = new ArrayList<>();

        // Pick half of the population size
        for (int i = 0; i < super.indNumber / 2; i++) {

            Collections.shuffle(patches, super.individualRng);

            // Best patch from x% randomly selected patches picked each time
            Patch bestPatch = patches.get(0);
            double best = population.get(bestPatch);
            for (int j = 1; j < (super.indNumber * TOURNAMENT_PERCENTAGE); j++) {
                Patch patch = patches.get(j);
                double fitness = population.get(patch);

                if (compareFitness(fitness, best) > 0) {
                    bestPatch = patch;
                    best = fitness;
                }
            }

            selectedPatches.add(bestPatch.clone());

        }
        return selectedPatches;
    }

    // Uniform crossover: patch1patch2 and patch2patch1 created, each edit added with x% probability
    protected List<Patch> crossover(List<Patch> patches, Patch origPatch) {

        List<Patch> crossedPatches = new ArrayList<>();

        Collections.shuffle(patches, super.individualRng);
        int half = patches.size() / 2;
        for (int i = 0; i < half; i++) {

            Patch parent1 = patches.get(i);
            Patch parent2 = patches.get(i + half);
            List<Edit> list1 = parent1.getEdits();
            List<Edit> list2 = parent2.getEdits();

            Patch child1 = origPatch.clone();
            Patch child2 = origPatch.clone();

            for (Edit edit : list1) {
                if (super.mutationRng.nextFloat() > MUTATE_PROBABILITY) {
                    child1.add(edit);
                }
            }
            for (Edit edit : list2) {
                if (super.mutationRng.nextFloat() > MUTATE_PROBABILITY) {
                    child1.add(edit);
                }
                if (super.mutationRng.nextFloat() > MUTATE_PROBABILITY) {
                    child2.add(edit);
                }
            }
            for (Edit edit : list1) {
                if (super.mutationRng.nextFloat() > MUTATE_PROBABILITY) {
                    child2.add(edit);
                }
            }

            crossedPatches.add(parent1);
            crossedPatches.add(parent2);
            crossedPatches.add(child1);
            crossedPatches.add(child2);
        }

        return crossedPatches;
    }

}
