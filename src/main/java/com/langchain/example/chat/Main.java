package com.langchain.example.chat;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;
import io.github.cdimascio.dotenv.Dotenv;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static dev.langchain4j.data.message.SystemMessage.systemMessage;
import static dev.langchain4j.data.message.UserMessage.userMessage;
import static java.util.Arrays.asList;

class KeyPoint{
    @Description("The key point in a few words")
    String keyPoint;

    @Description("The elaboration of the key point")
    String elaboration;
}

class KeyPoints {
    List<KeyPoint> keyPoints;
}

interface KeyPointExtractor{
    @UserMessage("Extract the key points from the following text: {{text}}")
    KeyPoints extractKeyPointsFrom(@V("text") String text);
}

@Slf4j
public class Main {

    public static void main(String[] args) {
        chat("why is the sky blue?");
        System.out.println();
        System.out.println("-".repeat(50));
        stream("why are leaves green?");
        System.out.println();
        System.out.println("-".repeat(50));
        String importantText = """
                Stretching keeps the muscles flexible and healthy, and we need that flexibility to maintain a range of 
                motion in the joints. Without it, the muscles shorten and become tight. Then, when you call on the 
                muscles for activity, they are unable to extend all the way. That puts you at risk for joint pain, 
                strains, and muscle damage.
                
                For example, sitting in a chair all day results in tight hamstrings in the back of the thigh. That can 
                make it harder to extend your leg or straighten your knee all the way, which inhibits walking. 
                Likewise, when tight muscles are suddenly called on for a strenuous activity that stretches them, such 
                as playing tennis, they may become damaged from suddenly being stretched. Injured muscles may not be 
                strong enough to support the joints, which can lead to joint injury.
                
                Regular stretching keeps muscles long, lean, and flexible, and this means that exertion won't put too 
                much force on the muscle itself. Healthy muscles also help a person with balance problems avoid falls.
                
                With a body full of muscles, the idea of daily stretching may seem overwhelming. It's most important to 
                focus on the body areas needed for critical for mobility: your lower extremities: your calves, your 
                hamstrings, your hip flexors in the pelvis and quadriceps in the front of the thigh. Stretching your 
                shoulders, neck, and lower back is also beneficial. Aim for a program of daily stretches, or at least 
                three or four times per week.
                """;
        getKeyPoints(importantText);
    }

    /**
     * Utility function to load OpenAI API key safely from
     * .env file
     * @return OpenAI API Key
     */
    public static String getApiKey(){
        Dotenv dotenv = Dotenv.load();
        return dotenv.get("OPENAI_API_KEY");
    }

    /**
     * Simple invocation of a query
     * @param query
     */
    public static void chat(String query){
        OpenAiChatModel model = OpenAiChatModel.builder()
                .apiKey(getApiKey())
                .modelName("gpt-5-nano")
                .build();
        System.out.println(model.chat(query));
    }

    /**
     * Stream invocation of a query
     * @param query
     */
    public static void stream(String query) {
        StreamingChatModel model = OpenAiStreamingChatModel.builder()
                .apiKey(getApiKey())
                .modelName("gpt-5-nano")
                .build();

        List<ChatMessage> messages = asList(
                systemMessage("You are a helpful assistant"),
                userMessage(query)
        );

        CompletableFuture<ChatResponse> futureResponse = new CompletableFuture<>();

        model.chat(messages, new StreamingChatResponseHandler() {
            @Override
            public void onCompleteResponse(ChatResponse chatResponse) {
                futureResponse.complete(chatResponse);
            }

            @Override
            public void onPartialResponse(String partialResponse) {
                System.out.print(partialResponse);
            }

            @Override
            public void onError(Throwable error) {
                futureResponse.completeExceptionally(error);
            }
        });

        futureResponse.join();
    }

    /**
     * Structured output. Extract key points from a string of text
     * @param text
     */
    public static void getKeyPoints(String text){
        KeyPointExtractor extractor = AiServices.create(
                KeyPointExtractor.class,
                OpenAiChatModel.builder()
                        .apiKey(getApiKey())
                        .modelName("gpt-5-nano")
                        .build()
        );
        KeyPoints result = extractor.extractKeyPointsFrom(text);

        result.keyPoints.forEach(
                kp -> System.out.println(kp.keyPoint + " - " + kp.elaboration)
        );
    }
}

