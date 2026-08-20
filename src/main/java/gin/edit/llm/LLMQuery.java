package gin.edit.llm;
import java.util.Map;

public interface LLMQuery {
    boolean testServerReachable();
    String chatLLM(String prompt);
    String chatLLMEnsemble(String prompt, Map<String, Object> format);
}
