package com.langchain.example.rag;


import com.openai.models.containers.files.content.ContentRetrieveParams;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class RAG {
    private InMemoryEmbeddingStore vectorDB;
    private OllamaEmbeddingModel embeddingModel;
    private OllamaChatModel llm;
    private ContentRetriever retriever;

    public RAG(String filePath, int maxResults, double minScore){
        this.embeddingModel = OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("mxbai-embed-large:latest")
                .build();
        this.llm = OllamaChatModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("deepseek-r1:8b")
                .build();

        // Chunk
        List<Document> documents = FileSystemDocumentLoader.loadDocuments(filePath);

        // Ingest
        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();
        ingestor.ingest(documents);

        this.vectorDB = embeddingStore;
        this.retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(this.embeddingModel)
                .maxResults(maxResults)
                .minScore(minScore)
                .build();
    }

    public RAG(){
        this(
                "/Users/tituslim/Documents/Personal Learning Folder/Personal Projects/LangChainJava/src/main/resources/harvard_oss_paper.pdf",
                4,
                0.5
        );
    }


}
