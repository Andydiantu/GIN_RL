package gin.edit.llm;

import gin.edit.llm.LLMConfig;
import gin.edit.llm.PromptTemplate;
import gin.edit.llm.PromptTemplate.PromptTag;

import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.Node;
import gin.SourceFileTree;
import gin.edit.Edit;
import gin.edit.statement.CopyStatement;
import gin.edit.statement.ReplaceStatement;
import gin.edit.statement.SwapStatement;

import org.json.JSONObject;
import org.json.JSONException;
import org.pmw.tinylog.Logger;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class LLMTargetSelector {

    LLMQuery llmQuery;

    public LLMTargetSelector() {
        if (LLMConfig.isOpenAICompatibleModelType()) {
            llmQuery = new OpenAINativeLLMQuery();
        } else {
            llmQuery = new Ollama4jLLMQuery("http://localhost:11434", LLMConfig.modelType);
        }
    }

    public String getStatementIDsAndNodes(Node node) {
        Map<Integer, Statement> statementMap = new TreeMap<>();

        List<Statement> statements = node.findAll(Statement.class);

        for (Statement stmt : statements) {
            Integer id = stmt.containsData(SourceFileTree.NODEKEY_ID) ? stmt.getData(SourceFileTree.NODEKEY_ID)
                    : SourceFileTree.NODE_NULL_ID;
            statementMap.put(id, stmt);
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Integer, Statement> entry : statementMap.entrySet()) {
            Integer id = entry.getKey();
            Statement stmt = entry.getValue();
            sb.append(id).append(" -> ").append(stmt.toString()).append("\n");
        }
        return sb.toString();
    }

    /**
     * Select target statement ID using the appropriate template for the given edit
     * class
     * 
     * @param targetMethodRootNode The AST node to analyze
     * @param editClass            The class of edit operation being performed
     * @return The selected statement ID
     */
    public int selectTargetSingle(Node targetMethodRootNode, Class<? extends Edit> editClass) {
        String statementIDsAndNodes = getStatementIDsAndNodes(targetMethodRootNode);

        PromptTemplate promptTemplate = LLMConfig.getPromptTemplateForEdit(editClass);
        Map<String, Object> format = LLMConfig.getFormatForEdit(editClass);
        Map<PromptTag, String> tagReplacements = new HashMap<>();
        tagReplacements.put(PromptTag.AST, statementIDsAndNodes);
        tagReplacements.put(PromptTag.DESTINATION, targetMethodRootNode.toString());

        String prompt = promptTemplate.replaceTags(tagReplacements);

        Logger.info("============");
        Logger.info("prompt for target selector:");
        Logger.info(prompt);
        Logger.info("Format: " + format.toString());
        Logger.info("============");

        String response = llmQuery.chatLLMEnsemble(prompt, format);
        Logger.info("============");
        Logger.info("response from target selector:");
        Logger.info(response);
        Logger.info("============");

        JSONObject jsonObject = new JSONObject(response);
        int statementID = jsonObject.getInt("StatementID");

        if (!isTargetValid(statementID, targetMethodRootNode)) {
            Logger.info("Invalid target ID: " + statementID);
            throw new IllegalArgumentException("Invalid target ID: " + statementID);
        }

        return statementID;

    }

    public int[] selectTargetMultiple(Node targetMethodRootNode, Class<? extends Edit> editClass) {
        String statementIDsAndNodes = getStatementIDsAndNodes(targetMethodRootNode);

        PromptTemplate promptTemplate = LLMConfig.getPromptTemplateForEdit(editClass);
        Map<String, Object> format = LLMConfig.getFormatForEdit(editClass);
        Map<PromptTag, String> tagReplacements = new HashMap<>();
        tagReplacements.put(PromptTag.AST, statementIDsAndNodes);
        tagReplacements.put(PromptTag.DESTINATION, targetMethodRootNode.toString());

        String prompt = promptTemplate.replaceTags(tagReplacements);

        Logger.info("============");
        Logger.info("prompt for target selector:");
        Logger.info(prompt);
        Logger.info("Format: " + format.toString());
        Logger.info("============");

        String response = llmQuery.chatLLMEnsemble(prompt, format);
        Logger.info("============");
        Logger.info("response from target selector:");
        Logger.info(response);
        Logger.info("============");

        if (editClass == CopyStatement.class) {
            JSONObject jsonObject = new JSONObject(response);
            int result1 = jsonObject.getInt("ingredientID");
            int result2 = jsonObject.getInt("targetBlockID");
            int result3 = jsonObject.getInt("anchorID");

            if (!isTargetValid(new int[] { result1, result2, result3 }, targetMethodRootNode)) {
                Logger.info("Invalid target IDs: " + new int[] { result1, result2, result3 });
                throw new IllegalArgumentException("Invalid target IDs: " + new int[] { result1, result2, result3 });
            }
            return new int[] { result1, result2, result3 };
        } else if (editClass == ReplaceStatement.class) {
            JSONObject jsonObject = new JSONObject(response);
            int result1 = jsonObject.getInt("TargetID");
            int result2 = jsonObject.getInt("ingredientID");

            if (!isTargetValid(new int[] { result1, result2 }, targetMethodRootNode)) {
                Logger.info("Invalid target IDs: " + new int[] { result1, result2 });
                throw new IllegalArgumentException("Invalid target IDs: " + new int[] { result1, result2 });
            }
            return new int[] { result1, result2 };
        } else {
            // For swap mutation
            JSONObject jsonObject = new JSONObject(response);
            int result1 = jsonObject.getInt("firstID");
            int result2 = jsonObject.getInt("secondID");

            if (!isTargetValid(new int[] { result1, result2 }, targetMethodRootNode)) {
                Logger.info("Invalid target IDs: " + new int[] { result1, result2 });
                throw new IllegalArgumentException("Invalid target IDs: " + new int[] { result1, result2 });
            }

            return new int[] { result1, result2 };
        }

    }

    public boolean isTargetValid(int targetStatementID, Node targetMethodRootNode) {
        List<Statement> statements = targetMethodRootNode.findAll(Statement.class);

        for (Statement stmt : statements) {
            Integer id = stmt.containsData(SourceFileTree.NODEKEY_ID) ? stmt.getData(SourceFileTree.NODEKEY_ID)
                    : SourceFileTree.NODE_NULL_ID;
            if (id == targetStatementID) {
                return true;
            }
        }
        return false;
    }

    public boolean isTargetValid(int[] targetStatementIDs, Node targetMethodRootNode) {
        for (int targetStatementID : targetStatementIDs) {
            if (!isTargetValid(targetStatementID, targetMethodRootNode)) {
                return false;
            }
        }
        return true;
    }

}