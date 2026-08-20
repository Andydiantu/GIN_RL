package gin.edit.llm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.pmw.tinylog.Logger;

import com.github.javaparser.ParseProblemException;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.Statement;

import gin.SourceFile;
import gin.SourceFileTree;
import gin.edit.Edit;
import gin.edit.llm.PromptTemplate.PromptTag;
import gin.edit.statement.StatementEdit;

public class LLMReplaceStatement extends StatementEdit {

    private static final long serialVersionUID = 1112502387236768006L;
    public String destinationFilename;
    public int destinationStatement;

    private PromptTemplate promptTemplate;

    /** fairly rubbish approach to having something meaningful for the toString */
    private String lastReplacement;
    private String lastPrompt;

    /**
     * true if this instance was constructed by a call to fromString()
     * In this case, generation state is loaded from serialized content.
     */
    private boolean recreatedFromString;

    private enum GenerationState {
        UNGENERATED,
        GENERATED_OK,
        GENERATED_EMPTY
    }

    private GenerationState generationState;
    private List<String> cachedReplacementStrings;

    /**
     * create a random llmreplacestatement for the given sourcefile, using the
     * provided RNG
     *
     * all this does is pick a location
     *
     * @param sourceFile to create an edit for
     * @param rng random number generator, used to choose the target statements
     */
    public LLMReplaceStatement(SourceFile sourceFile, Random rng, PromptTemplate promptTemplate) {
        SourceFileTree sf = (SourceFileTree) sourceFile;
        MethodDeclaration targetMethod = getTargetMethod(sf);

        destinationFilename = sourceFile.getRelativePathToWorkingDir();
        destinationStatement = getTargetMethodBodyID(targetMethod);

        this.promptTemplate = promptTemplate;

        lastReplacement = "NOT YET APPLIED";
        lastPrompt = "NOT YET APPLIED";
        recreatedFromString = false;
        generationState = GenerationState.UNGENERATED;
        cachedReplacementStrings = new ArrayList<>();
    }

    public LLMReplaceStatement(SourceFile sourceFile, Random rng) {
        // Always use the SM prompt template from Brownlee1 2025 for replace mutation
        this(sourceFile, rng, LLMConfig.PromptType.RUNTIME_OPTIMISATION_WITH_EXAMPLES.template);
    }

    public LLMReplaceStatement(String destinationFilename, int destinationStatement) {
        this.destinationFilename = destinationFilename;
        this.destinationStatement = destinationStatement;
        this.promptTemplate = LLMConfig.PromptType.RUNTIME_OPTIMISATION_WITH_EXAMPLES.template;

        this.lastPrompt = "NOT YET APPLIED";
        this.lastReplacement = "NOT YET APPLIED";
        this.recreatedFromString = false;
        this.generationState = GenerationState.UNGENERATED;
        this.cachedReplacementStrings = new ArrayList<>();
    }

    public static Edit fromString(String description) {

        // the following will give us 5 tokens:
        // gin.edit.llm.LLMReplaceStatement
        // src/main/java/org/apache/commons/net/smtp/SimpleSMTPHeader.java:331\nPrompt:
        // (prompt)
        // --->
        // (replacement)
        // ""
        String[] tokens1 = description.split("!!!", -1);

        // split the first of these to get the filename and statement ID
        String[] tokens2 = tokens1[0].split("\\s+(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
        String[] destTokens = tokens2[1].split(":");
        String destFilename = destTokens[0].replace("\"", "");
        int destination = Integer.parseInt(destTokens[1]);

        LLMReplaceStatement rval = new LLMReplaceStatement(destFilename, destination);
        // Unescape newlines that were escaped in toString() for CSV safety
        rval.lastReplacement = tokens1[3].replace("\\n", "\n").replace("\\r", "\r").trim();
        rval.lastPrompt = tokens1[1].replace("\\n", "\n").replace("\\r", "\r").trim();
        rval.recreatedFromString = true;

        if ("LLM GAVE NO SUGGESTIONS".equals(rval.lastReplacement)) {
            rval.generationState = GenerationState.GENERATED_EMPTY;
            rval.cachedReplacementStrings = new ArrayList<>();
        } else {
            rval.generationState = GenerationState.GENERATED_OK;
            rval.cachedReplacementStrings = new ArrayList<>(Collections.singletonList(rval.lastReplacement));
        }

        return rval;
    }

    protected LLMQuery createQuery() {
        if (LLMConfig.isOpenAICompatibleModelType()) {
            return new OpenAINativeLLMQuery();
        }
        return new Ollama4jLLMQuery("http://localhost:11434", LLMConfig.modelType);
    }

    @Override
    public SourceFile apply(SourceFile sourceFile, Object tagReplacements) {
        List<SourceFile> l = applyMultiple(sourceFile, 5, (Map<PromptTemplate.PromptTag, String>) tagReplacements);

        if (l.size() > 0) {
            return l.get(0); // TODO for now, just pick the first variant provided. Later, call applyMultiple
                             // from LocalSearch instead
        } else {
            return null;
        }
    }

    public List<SourceFile> applyMultiple(SourceFile sourceFile, int count,
            Map<PromptTemplate.PromptTag, String> tagReplacements) {

        SourceFileTree sf = (SourceFileTree) sourceFile;
        MethodDeclaration targetMethod = getTargetMethod(sf);

        Node destination = sf.getNode(destinationStatement);

        if (destination == null) {
            return Collections.singletonList(sf); // targeting a deleted location just does nothing.
        }

        if (generationState == GenerationState.UNGENERATED) {
            generateReplacements(targetMethod, destination, count, tagReplacements);
        }

        return applyCachedReplacements(sf);
    }

    private void generateReplacements(MethodDeclaration targetMethod, Node destination, int count,
            Map<PromptTemplate.PromptTag, String> tagReplacements) {
        LLMQuery llmQuery = createQuery();

        Logger.info("Seeking replacements for:");
        Logger.info(destination);

        if (tagReplacements == null) {
            tagReplacements = new HashMap<>();
        }

        tagReplacements.put(PromptTag.COUNT, Integer.toString(count));
        tagReplacements.put(PromptTag.DESTINATION, targetMethod.getDeclarationAsString(true, true, true) + " "
                + targetMethod.getBody()
                .orElseThrow(() -> new IllegalStateException("Target method has no body.")).toString());
        tagReplacements.put(PromptTag.PROJECT, LLMConfig.projectName);

        if (promptTemplate == null) {
            promptTemplate = LLMConfig.PromptType.RUNTIME_OPTIMISATION_WITH_EXAMPLES.template;
        }

        String prompt = promptTemplate.replaceTags(tagReplacements);

        Logger.info("============");
        Logger.info("prompt:");
        Logger.info(prompt);
        lastPrompt = prompt;
        Logger.info("============");

        String answer = llmQuery.chatLLM(prompt);

        Pattern pattern = Pattern.compile("```(?:java)(.*?)```", Pattern.DOTALL | Pattern.MULTILINE);
        Matcher matcher = pattern.matcher(answer);

        List<String> replacementStrings = new ArrayList<>();

        while (matcher.find()) {
            String str = matcher.group(1);

            try {
                StaticJavaParser.parseBlock(str);
                replacementStrings.add(str);
            } catch (Throwable e) {
                if (!(e instanceof ParseProblemException)) {
                    Logger.info("Unexpected error parsing LLM response: " + e.getClass().getName() + " - "
                            + e.getMessage());
                }
            }
        }

        int i = 1;
        for (String s : replacementStrings) {
            Logger.info("============");
            Logger.info("suggestion " + i++);
            Logger.info(s);
            Logger.info("============");
        }

        if (replacementStrings.isEmpty()) {
            Logger.info("============");
            Logger.info("No replacements found. Response was:");
            Logger.info(answer);
            Logger.info("============");
            lastReplacement = "LLM GAVE NO SUGGESTIONS";
            generationState = GenerationState.GENERATED_EMPTY;
            cachedReplacementStrings = new ArrayList<>();
            return;
        }

        lastReplacement = replacementStrings.get(0);
        generationState = GenerationState.GENERATED_OK;
        cachedReplacementStrings = replacementStrings;
    }

    private List<SourceFile> applyCachedReplacements(SourceFileTree sf) {
        if (generationState == GenerationState.GENERATED_EMPTY) {
            return new ArrayList<>();
        }

        List<SourceFile> variantSourceFiles = new ArrayList<>();

        for (String replacementString : cachedReplacementStrings) {
            try {
                Statement stmt = StaticJavaParser.parseBlock(replacementString);
                variantSourceFiles.add(sf.replaceNode(destinationStatement, stmt));
            } catch (ClassCastException e) {
                // JavaParser sometimes throws this if the statements don't match
            } catch (Throwable e) {
                Logger.error("Problem parsing cached edit: " + replacementString + " - " + e.getClass().getName() + ": "
                        + e.getMessage());
            }
        }

        return variantSourceFiles;
    }

    private static MethodDeclaration getTargetMethod(SourceFileTree sf) {
        List<Node> targetMethodNodes = sf.getTargetMethodRootNode();
        if (targetMethodNodes == null || targetMethodNodes.isEmpty()) {
            throw new IllegalStateException("No target method found for LLMReplaceStatement.");
        }
        if (!(targetMethodNodes.get(0) instanceof MethodDeclaration)) {
            throw new IllegalStateException("Target node is not a method declaration.");
        }
        return (MethodDeclaration) targetMethodNodes.get(0);
    }

    private static int getTargetMethodBodyID(MethodDeclaration targetMethod) {
        BlockStmt body = targetMethod.getBody()
                .orElseThrow(() -> new IllegalStateException("Target method has no body."));
        if (!body.containsData(SourceFileTree.NODEKEY_ID)) {
            throw new IllegalStateException("Target method body has no node ID.");
        }
        return body.getData(SourceFileTree.NODEKEY_ID);
    }

    @Override
    public String toString() {
        // Escape newlines to prevent breaking CSV format
        String safePrompt = (lastPrompt != null) ? lastPrompt.replace("\n", "\\n").replace("\r", "\\r") : "null";
        String safeReplacement = (lastReplacement != null) ? lastReplacement.replace("\n", "\\n").replace("\r", "\\r") : "null";
        return this.getClass().getCanonicalName() + " \"" + destinationFilename + "\":" + destinationStatement
                + " Prompt: !!! " + safePrompt + " !!! --> !!! " + safeReplacement + " !!!";
    }

}
