package com.langchain.example.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.FileSystems;
import java.nio.file.PathMatcher;
import java.time.Duration;
import java.util.List;

interface Assistant {
    String chat(String userMessage);
}

@Slf4j
public class RAG {
    private OllamaEmbeddingModel embeddingModel;
    private OllamaChatModel llm;
    private ContentRetriever retriever;
    private Assistant assistant;

    public RAG(String directoryPath, int maxResults, double minScore){
        this.embeddingModel = OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("mxbai-embed-large:latest")
                .build();
        this.llm = OllamaChatModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("qwen2.5:latest")
                .timeout(Duration.ofMinutes(10))
                .build();

        // Chunk
        System.out.println("Chunking documents");
        PathMatcher pathMatcher = FileSystems.getDefault().getPathMatcher("glob:*.pdf");
        DocumentParser pdfParser = new ApachePdfBoxDocumentParser();
        List<Document> documents = FileSystemDocumentLoader.loadDocuments(directoryPath, pathMatcher, pdfParser);

        // Ingest
        System.out.println("Generating embeddings");
        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();
        System.out.println("Ingesting embeddings");
        ingestor.ingest(documents);

        this.retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(this.embeddingModel)
                .maxResults(maxResults)
                .minScore(minScore)
                .build();
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(retriever)
                .build();
        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(llm)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

    public RAG(){
        this(
                "src/main/resources",
                4,
                0.5
        );
    }

    public String askChatbot(String query){
        System.out.printf("Received a query: %s \nGenerating response...%n", query);
        return assistant.chat(query);
    }
}
