package gin.edit.llm;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import io.github.ollama4j.Ollama;
import io.github.ollama4j.exceptions.OllamaException;
import io.github.ollama4j.models.chat.OllamaChatMessageRole;
import io.github.ollama4j.models.chat.OllamaChatRequest;
import io.github.ollama4j.models.chat.OllamaChatResult;
import io.github.ollama4j.models.generate.OllamaGenerateRequest;
import io.github.ollama4j.models.request.ThinkMode;
import io.github.ollama4j.models.response.OllamaResult;

import gin.edit.llm.LLMQuery;

import org.pmw.tinylog.Logger;

public class Ollama4jLLMQuery implements LLMQuery {
    private Ollama ollama;
    private String modelType;
    private List<String> modelTypes;

    // c'tor
    public Ollama4jLLMQuery(String ollamaServerHost, String modelType) {

        if (modelType.contains(",")) {
            this.modelTypes = Arrays.asList(modelType.split(","));
            this.modelType = modelTypes.get(0);
        } else {
            this.modelTypes = Collections.singletonList(modelType);
            this.modelType = modelType;
        }

        this.ollama = new Ollama(ollamaServerHost);
        ollama.setRequestTimeoutSeconds(LLMConfig.timeoutInSeconds);
    }

   @Override
    public boolean testServerReachable() {
        try {
            return ollama.ping();
        } catch (OllamaException e) {
            e.printStackTrace();
            return false;
        }
    }

    @Override
    public String chatLLM(String prompt) {
        try {
            OllamaGenerateRequest request = OllamaGenerateRequest.builder()
                .withModel(modelType)
                .withThink(ThinkMode.DISABLED)
                .withPrompt(prompt)
                .withKeepAlive("1h")
                .build();

            OllamaResult result = ollama.generate(request, null);
            return result.getResponse();
        } catch (OllamaException e) {
            e.printStackTrace();
            Logger.error("Error from LLM: " + e.getMessage());
        }
        return "";
    }

    public String chatLLMEnsemble(String prompt, Map<String, Object> format) {
        String targetModelType = null;
        if (modelTypes.size() > 1) {
            // Randomly select a model from modelTypes with uniform probability
            Random rng = new Random();
            int randomIndex = rng.nextInt(modelTypes.size());
            targetModelType = modelTypes.get(randomIndex);
        } else {
            targetModelType = modelType;
        }

        Logger.info("Using model: " + targetModelType + "chosen from " + modelTypes.toString());

        // Map<String, Object> StatementID = new HashMap<>();
        // StatementID.put("type", Integer.class.getSimpleName().toLowerCase());

        // Map<String, Object> properties = new HashMap<>();
        // properties.put("StatementID", StatementID);

        // Map<String, Object> format = new HashMap<>();
        // format.put("type", "object");
        // format.put("properties", properties);
        // format.put("required", Arrays.asList("StatementID"));

        Logger.info("Format: " + format.toString());

        try {
            OllamaGenerateRequest request = null;
            if (format != null) {
                request = OllamaGenerateRequest.builder()
                    .withModel(targetModelType)
                    .withPrompt(prompt)
                    .withFormat(format)
                    .withThink(ThinkMode.DISABLED)
                    .withKeepAlive("1h")
                    .build();
            }
            else {
                request = OllamaGenerateRequest.builder()
                    .withModel(targetModelType)
                    .withPrompt(prompt)
                    .withThink(ThinkMode.DISABLED)
                    .withKeepAlive("1h")
                    .build();
            }

            OllamaResult result = ollama.generate(request, null);

            Logger.info("Response from LLM: " + result.getResponse());
            return result.getResponse();
        } catch (OllamaException e) {
            e.printStackTrace();
            Logger.error("Error from LLM: " + e.getMessage());
        }
        return "";
    }
    
}

