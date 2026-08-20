package gin.edit.llm;

import java.util.ArrayList;
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
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.EmptyStmt;
import com.github.javaparser.ast.stmt.Statement;

import gin.SourceFile;
import gin.SourceFileTree;
import gin.edit.llm.PromptTemplate.PromptTag;
import gin.edit.statement.StatementEdit;

public class LLMMaskedStatement extends StatementEdit {
    private static final long serialVersionUID = 1112502387236768004L;
    public String destinationFilename;
    public int destinationStatement = -1;

    private PromptTemplate promptTemplate;

    private String lastReplacement;
    private String lastPrompt;

    private Random rng = null;

    private enum GenerationState {
        UNGENERATED,
        GENERATED_OK,
        GENERATED_EMPTY
    }

    private GenerationState generationState;
    private List<String> cachedReplacementStrings;

    public LLMMaskedStatement(SourceFile sourceFile, Random rng, PromptTemplate promptTemplate) {
        SourceFileTree sf = (SourceFileTree) sourceFile;

        destinationFilename = sourceFile.getRelativePathToWorkingDir();

        destinationStatement = sf.getRandomBlockID(true, rng);

        this.promptTemplate = promptTemplate;

        this.rng = rng;

        this.lastReplacement = "NOT YET APPLIED";
        this.lastPrompt = "NOT YET APPLIED";
        this.generationState = GenerationState.UNGENERATED;
        this.cachedReplacementStrings = new ArrayList<>();
    }

    public LLMMaskedStatement(SourceFile sourceFile, Random rng) {
        this(sourceFile, rng, LLMConfig.getDefaultPromptTemplate());
    }

    public LLMMaskedStatement(String destinationFilename, int destinationStatement) {
        this.destinationFilename = destinationFilename;
        this.destinationStatement = destinationStatement;
        this.promptTemplate = LLMConfig.getDefaultPromptTemplate();
        this.rng = new Random();
        this.lastPrompt = "NOT YET APPLIED";
        this.lastReplacement = "NOT YET APPLIED";
        this.generationState = GenerationState.UNGENERATED;
        this.cachedReplacementStrings = new ArrayList<>();
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

        Statement statementToMask = resolveStatementToMask(sf);
        if (statementToMask == null) {
            return new ArrayList<>();
        }

        if (generationState == GenerationState.UNGENERATED) {
            generateReplacements(sf, statementToMask, count, tagReplacements);
        }

        return applyCachedReplacements(sf);
    }

    private Statement resolveStatementToMask(SourceFileTree sf) {
        Statement statementToMask;

        if (destinationStatement == -1) {
            statementToMask = drawStatementFromSourceFile(sf, (rng != null ? rng : new Random()));
            if (statementToMask != null) {
                Integer id = statementToMask.containsData(SourceFileTree.NODEKEY_ID)
                        ? statementToMask.getData(SourceFileTree.NODEKEY_ID)
                        : SourceFileTree.NODE_NULL_ID;
                if (id != null && id != SourceFileTree.NODE_NULL_ID) {
                    destinationStatement = id;
                }
            }
        } else {
            statementToMask = sf.getStatement(destinationStatement);
        }

        if (statementToMask == null) {
            Logger.warn("No target statement found for mask edit. destinationStatement=" + destinationStatement);
            return null;
        }

        Logger.info("Statement to mask: " + statementToMask);
        return statementToMask;
    }

    private void generateReplacements(SourceFileTree sf, Statement statementToMask, int count,
            Map<PromptTemplate.PromptTag, String> tagReplacements) {
        LLMQuery llmQuery = createQuery();

        if (tagReplacements == null) {
            tagReplacements = new HashMap<>();
        }

        tagReplacements.put(PromptTag.PROJECT, LLMConfig.projectName);
        tagReplacements.put(PromptTag.COUNT, Integer.toString(count));
        tagReplacements.put(PromptTag.DESTINATION, maskCode(sf, statementToMask));
        tagReplacements.put(PromptTag.ORIGINAL_CODE, statementToMask.toString());

        // Add context information for enhanced prompts
        tagReplacements.put(PromptTag.METHOD_SIGNATURE, extractMethodSignature(sf));
        tagReplacements.put(PromptTag.CLASS_FIELDS, extractClassFields(sf));
        tagReplacements.put(PromptTag.IMPORTS, extractImports(sf));
        tagReplacements.put(PromptTag.LOCAL_VARIABLES, extractLocalVariables(sf, destinationStatement));

        if (promptTemplate == null) {
            promptTemplate = LLMConfig.getDefaultPromptTemplate();
        }

        String prompt = promptTemplate.replaceTags(tagReplacements);

        Logger.info("============");
        Logger.info("prompt:");
        Logger.info(prompt);
        lastPrompt = prompt;
        Logger.info("============");

        String answer = llmQuery.chatLLM(prompt);

        Logger.info("============");
        Logger.info("response:");
        Logger.info(answer);
        Logger.info("============");

        // extract code blocks (both ```java ... ``` and plain ``` ... ```)
        Pattern pattern = Pattern.compile("```(?:java)?\\s*([\\s\\S]*?)```", Pattern.DOTALL | Pattern.MULTILINE);
        Matcher matcher = pattern.matcher(answer);

        List<String> replacementStrings = new ArrayList<>();

        while (matcher.find()) {
            String str = matcher.group(1);

            Logger.info("============");
            Logger.info("match:");
            Logger.info(str);

            String trimmedStr = str.trim();
            if (trimmedStr.isEmpty()) {
                Logger.info("Empty code block; skipping.");
                continue;
            }

            if (trimmedStr.matches("(?s).*//\\s*\\.{3}\\s*\\}\\s*$")
                    || trimmedStr.matches("(?s)^\\s*//\\s*\\.{3}\\s*$")
                    || trimmedStr.equals("// ...")
                    || trimmedStr.equals("// ...\\n}")
                    || trimmedStr.endsWith("// ...\\n}")) {
                Logger.info("Code contains placeholder ellipsis pattern (// ...); skipping.");
                continue;
            }

            if (parseCandidateStatement(trimmedStr) == null) {
                continue;
            }

            replacementStrings.add(str);
            Logger.info("============");
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
        Logger.info("============");
        Logger.info("Applying first suggestion:");
        Logger.info(lastReplacement);
        Logger.info("============");

        generationState = GenerationState.GENERATED_OK;
        cachedReplacementStrings = replacementStrings;
    }

    private List<SourceFile> applyCachedReplacements(SourceFileTree sf) {
        if (generationState == GenerationState.GENERATED_EMPTY) {
            return new ArrayList<>();
        }

        List<SourceFile> variantSourceFiles = new ArrayList<>();

        for (String replacementString : cachedReplacementStrings) {
            Statement parsed = parseCandidateStatement(replacementString.trim());
            if (parsed == null) {
                continue;
            }

            try {
                variantSourceFiles.add(sf.replaceNode(destinationStatement, parsed));
            } catch (ClassCastException e) {
                // JavaParser sometimes throws this if the statements don't match
            }
        }

        return variantSourceFiles;
    }

    private Statement parseCandidateStatement(String candidate) {
        try {
            Statement stmt = StaticJavaParser.parseStatement(candidate);
            Logger.info("Parsed as single statement.");
            return stmt;
        } catch (Throwable eStmt) {
            if (!(eStmt instanceof ParseProblemException)) {
                Logger.info("Unexpected error parsing as statement: " + eStmt.getClass().getName() + " - "
                        + eStmt.getMessage());
            }
            Logger.info("Failed to parse as single statement, trying as block.");
        }

        try {
            Statement stmt = StaticJavaParser.parseBlock(candidate);
            Logger.info("Parsed as block.");
            return stmt;
        } catch (Throwable eBlock) {
            if (!(eBlock instanceof ParseProblemException)) {
                Logger.info("Unexpected error parsing as block: " + eBlock.getClass().getName() + " - "
                        + eBlock.getMessage());
            }
            Logger.info("Failed to parse as block, trying as method declaration.");
        }

        try {
            MethodDeclaration method = StaticJavaParser.parseMethodDeclaration(candidate);
            Statement stmt = method.getBody().orElse(null);
            if (stmt != null) {
                Logger.info("Parsed as method declaration; using body.");
                return stmt;
            }
            Logger.info("Method declaration had no body; skipping.");
            return null;
        } catch (Throwable eMethod) {
            if (!(eMethod instanceof ParseProblemException)) {
                Logger.info("Unexpected error parsing as method: " + eMethod.getClass().getName() + " - "
                        + eMethod.getMessage());
            }
            Logger.info("Failed to parse as method declaration; skipping this match.");
            if (eMethod instanceof ParseProblemException) {
                Logger.info(eMethod);
            }
            return null;
        }
    }

    public Statement drawStatementFromSourceFile(SourceFileTree sourceFileTree, Random rng) {
        List<Statement> stmts = sourceFileTree.getTargetMethodRootNode().get(0).findAll(Statement.class);

        Statement stmt = stmts.get(rng.nextInt(stmts.size()));
        while (ifNonImpactfulStatement(stmt) && stmts.size() > 1) {
            Logger.info("Non-impactful statement found, trying another one, the statement is: " + stmt.toString());
            stmts.remove(stmt);
            stmt = stmts.get(rng.nextInt(stmts.size()));
        }
        return stmt;
    }

    public boolean ifNonImpactfulStatement(Statement stmt) {
        // TODO check if the statement is non-impactful
        if (stmt.isExpressionStmt() ||
                stmt.isBlockStmt() ||
                stmt.isForStmt() ||
                stmt.isIfStmt() ||
                stmt.isReturnStmt() ||
                stmt.isWhileStmt() ||
                stmt.isThrowStmt()) {
            return false;
        }
        return true;
    }

    public String maskCode(SourceFileTree sf, Statement targetStatement) {

        Logger.info("Target statement: " + targetStatement.toString());
        Logger.info(sf.getTargetMethodRootNode().get(0).toString());

        Node targetMethodRootNode = sf.getTargetMethodRootNode().get(0).clone();

        List<Statement> stmts = targetMethodRootNode.findAll(Statement.class);

        Statement stmtInClone = null;
        for (Statement s : stmts) {
            if (s.toString().equals(targetStatement.toString())) {
                stmtInClone = s;
                break;
            }
        }

        if (stmtInClone != null) {
            Statement placeholderStatement;
            if (stmtInClone instanceof BlockStmt) {
                placeholderStatement = new BlockStmt();
                placeholderStatement.setComment(new LineComment("<<PLACEHOLDER>>"));
            } else {
                placeholderStatement = new EmptyStmt();
                placeholderStatement.setComment(new LineComment("<<PLACEHOLDER>>"));
            }

            boolean ifReplaceSuc = stmtInClone.replace(placeholderStatement);

            if (!ifReplaceSuc) {
                Logger.error("Failed to replace the statement with placeholder");
            }
        } else {
            Logger.error("Could not find target statement in cloned tree");
        }

        String maskedCode = targetMethodRootNode.toString();

        Logger.info("============");
        Logger.info("masked code:");
        Logger.info(maskedCode);
        Logger.info("============");

        return maskedCode;
    }

    /**
     * Extract method signature from the target method
     */
    private String extractMethodSignature(SourceFileTree sf) {
        try {
            Node targetMethodRootNode = sf.getTargetMethodRootNode().get(0);

            // Find the MethodDeclaration containing this node
            Node current = targetMethodRootNode;
            while (current != null && !(current instanceof MethodDeclaration)) {
                current = current.getParentNode().orElse(null);
            }

            if (current instanceof MethodDeclaration) {
                MethodDeclaration method = (MethodDeclaration) current;
                return method.getDeclarationAsString();
            }
        } catch (Exception e) {
            Logger.warn("Could not extract method signature: " + e.getMessage());
        }
        return "// Method signature not available";
    }

    /**
     * Extract class-level field declarations
     */
    private String extractClassFields(SourceFileTree sf) {
        try {
            StringBuilder fields = new StringBuilder();
            Node targetMethodRootNode = sf.getTargetMethodRootNode().get(0);

            // Navigate up to find the CompilationUnit
            Node current = targetMethodRootNode;
            while (current != null && !(current instanceof CompilationUnit)) {
                current = current.getParentNode().orElse(null);
            }

            if (current instanceof CompilationUnit) {
                CompilationUnit cu = (CompilationUnit) current;

                // Get all field declarations from classes in this compilation unit
                List<FieldDeclaration> fieldDeclarations = cu.findAll(FieldDeclaration.class);

                if (fieldDeclarations.isEmpty()) {
                    return "// No class fields declared";
                }

                for (FieldDeclaration field : fieldDeclarations) {
                    fields.append(field.toString().trim()).append("\n");
                }

                return fields.toString().trim();
            }
        } catch (Exception e) {
            Logger.warn("Could not extract class fields: " + e.getMessage());
        }
        return "// Class fields not available";
    }

    /**
     * Extract import statements from the compilation unit
     */
    private String extractImports(SourceFileTree sf) {
        try {
            StringBuilder imports = new StringBuilder();
            Node targetMethodRootNode = sf.getTargetMethodRootNode().get(0);

            // Navigate up to find the CompilationUnit
            Node current = targetMethodRootNode;
            while (current != null && !(current instanceof CompilationUnit)) {
                current = current.getParentNode().orElse(null);
            }

            if (current instanceof CompilationUnit) {
                CompilationUnit cu = (CompilationUnit) current;
                List<ImportDeclaration> importDeclarations = cu.getImports();

                if (importDeclarations.isEmpty()) {
                    return "// No imports";
                }

                for (ImportDeclaration imp : importDeclarations) {
                    imports.append(imp.toString().trim()).append("\n");
                }

                return imports.toString().trim();
            }
        } catch (Exception e) {
            Logger.warn("Could not extract imports: " + e.getMessage());
        }
        return "// Imports not available";
    }

    /**
     * Extract local variables in scope for the target statement
     */
    private String extractLocalVariables(SourceFileTree sf, int statementID) {
        try {
            List<SourceFileTree.VariableTypeAndName> variables = sf
                    .getPrimitiveVariablesInScopeForStatement(statementID);

            if (variables == null || variables.isEmpty()) {
                return "// No local variables in scope";
            }

            StringBuilder localVars = new StringBuilder();
            for (SourceFileTree.VariableTypeAndName var : variables) {
                localVars.append(var.getType().toString())
                        .append(" ")
                        .append(var.getName().toString())
                        .append("\n");
            }

            return localVars.toString().trim();
        } catch (Exception e) {
            Logger.warn("Could not extract local variables: " + e.getMessage());
        }
        return "// Local variables not available";
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
