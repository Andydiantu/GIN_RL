package gin.edit.llm;

import java.time.Duration;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.core.JsonValue;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;

import org.pmw.tinylog.Logger;

public class OpenAINativeLLMQuery implements LLMQuery {
    private OpenAIClient client;
    private String modelName;
    private List<String> modelNames;

    private static boolean isDeepSeek() {
        return "DeepSeek".equalsIgnoreCase(LLMConfig.modelType);
    }

    private static String providerLabel() {
        return isDeepSeek() ? "DeepSeek" : "OpenAI";
    }

    /** Attach DeepSeek-specific body fields (thinking, reasoning_effort). No-op for OpenAI. */
    private static void applyDeepSeekThinking(ChatCompletionCreateParams.Builder builder) {
        if (!isDeepSeek()) {
            return;
        }
        Map<String, Object> thinking = new java.util.HashMap<>();
        thinking.put("type", LLMConfig.deepSeekThinking ? "enabled" : "disabled");
        builder.putAdditionalBodyProperty("thinking", JsonValue.from(thinking));
        if (LLMConfig.deepSeekThinking
                && LLMConfig.deepSeekReasoningEffort != null
                && !LLMConfig.deepSeekReasoningEffort.trim().isEmpty()) {
            builder.putAdditionalBodyProperty("reasoning_effort",
                JsonValue.from(LLMConfig.deepSeekReasoningEffort));
        }
    }

    /** Read message.content; if empty and DeepSeek returned reasoning_content (thinking mode), fall back to it. */
    private static String extractContent(ChatCompletion completion) {
        if (completion.choices() == null || completion.choices().isEmpty()) {
            return "";
        }
        com.openai.models.chat.completions.ChatCompletionMessage msg = completion.choices().get(0).message();
        String content = msg.content().orElse("");
        if (!content.isEmpty()) {
            return content;
        }
        // DeepSeek puts the chain-of-thought in `reasoning_content` (not part of the OpenAI schema).
        try {
            JsonValue extra = msg._additionalProperties().get("reasoning_content");
            if (extra != null) {
                String reasoning = extra.convert(String.class);
                if (reasoning != null && !reasoning.isEmpty()) {
                    return reasoning;
                }
            }
        } catch (Exception ignored) {
            // best-effort; if the SDK shape differs, just return empty content
        }
        return "";
    }

    // c'tor
    public OpenAINativeLLMQuery() {
        this(isDeepSeek() ? LLMConfig.deepSeekModelName : LLMConfig.openAIModelName);
    }

    public OpenAINativeLLMQuery(String modelName) {
        if (modelName.contains(",")) {
            this.modelNames = Arrays.asList(modelName.split(","));
            this.modelName = modelNames.get(0);
        } else {
            this.modelNames = Collections.singletonList(modelName);
            this.modelName = modelName;
        }

        String apiKey = isDeepSeek() ? LLMConfig.deepSeekKey : LLMConfig.openAIKey;
        String baseUrl = isDeepSeek() ? LLMConfig.deepSeekBaseUrl : null;

        OpenAIOkHttpClient.Builder clientBuilder = OpenAIOkHttpClient.builder()
            .fromEnv()
            .timeout(Duration.ofSeconds(LLMConfig.timeoutInSeconds));

        if (apiKey != null && !apiKey.trim().isEmpty()) {
            clientBuilder.apiKey(apiKey);
        }

        if (baseUrl != null && !baseUrl.trim().isEmpty()) {
            clientBuilder.baseUrl(baseUrl);
        }

        this.client = clientBuilder.build();

        Logger.info("OpenAINativeLLMQuery initialized with provider: " + providerLabel()
            + ", model: " + this.modelName
            + ", baseUrl: " + (baseUrl == null || baseUrl.trim().isEmpty() ? "(default)" : baseUrl)
            + ", timeout: " + LLMConfig.timeoutInSeconds + "s");
    }

    @Override
    public boolean testServerReachable() {
        // Cloud is generally always reachable, but we can do a simple ping
        try {
            ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                    .addUserMessage("ping")
                    .model(modelName)
                    .maxCompletionTokens(1)
                    .build();
            client.chat().completions().create(params);
            return true;
        } catch (Exception e) {
            Logger.error(providerLabel() + " server unreachable: " + e.getMessage());
            return false;
        }
    }

    @Override
    public String chatLLM(String prompt) {
        try {
                Logger.info("[OpenAINative] Sending request to model: " + modelName + " (" + providerLabel() + ")");
            ChatCompletionCreateParams.Builder builder = ChatCompletionCreateParams.builder()
                    .addUserMessage(prompt)
                    .model(modelName)
                    .temperature(LLMConfig.temperature);
            applyDeepSeekThinking(builder);
            ChatCompletionCreateParams params = builder.build();

                Logger.info("[OpenAINative] Calling " + providerLabel() + " API...");
            ChatCompletion completion = client.chat().completions().create(params);
                Logger.info("[OpenAINative] Received response from " + providerLabel());

            return extractContent(completion);
        } catch (Exception e) {
            e.printStackTrace();
            Logger.error("Error from " + providerLabel() + ": " + e.getMessage());
        }
        return "";
    }

    @Override
    public String chatLLMEnsemble(String prompt, Map<String, Object> format) {
        String targetModelName = null;
        if (modelNames.size() > 1) {
            // Randomly select a model from modelNames with uniform probability
            Random rng = new Random();
            int randomIndex = rng.nextInt(modelNames.size());
            targetModelName = modelNames.get(randomIndex);
        } else {
            targetModelName = modelName;
        }

        Logger.info("Using model: " + targetModelName + " chosen from " + modelNames.toString()
            + " (" + providerLabel() + ")");

        // Note: The official OpenAI client doesn't support arbitrary format maps
        // in the same way Ollama does. For structured output, you would use
        // response_format with JSON schema. For now, we ignore the format parameter.
        if (format != null) {
            Logger.info("Format: " + format.toString() + " (Note: format parameter not fully supported in native client)");
        }

        try {
            String effectivePrompt = prompt;
            if (format != null) {
                effectivePrompt = prompt + "\n\nReturn valid json only.";
            }

            ChatCompletionCreateParams.Builder paramsBuilder = ChatCompletionCreateParams.builder()
                    .addUserMessage(effectivePrompt)
                    .model(targetModelName)
                    .temperature(LLMConfig.temperature);

            if (format != null) {
                Map<String, Object> responseFormat = new java.util.HashMap<>();
                responseFormat.put("type", "json_object");
                paramsBuilder.putAdditionalBodyProperty("response_format", JsonValue.from(responseFormat));
            }

            applyDeepSeekThinking(paramsBuilder);
            ChatCompletionCreateParams params = paramsBuilder.build();

            ChatCompletion completion = client.chat().completions().create(params);

            String response = extractContent(completion);

            Logger.info("Response from " + providerLabel() + ": " + response);
            return response;
        } catch (Exception e) {
            e.printStackTrace();
            Logger.error("Error from " + providerLabel() + ": " + e.getMessage());
        }
        return "";
    }
}