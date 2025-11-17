package com.langchain.example.rag;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        RAG chatbot = new RAG();
        Scanner scanner = new Scanner(System.in);
        System.out.println("Hello! Ask me any question on the GEPA framework! Say 'end' to end the conversation");
        while (true){
            String query = scanner.nextLine();
            if (query.equalsIgnoreCase("end")){
                System.out.println("Goodbye!");
                break;
            }
            String response = chatbot.askChatbot(query);
            System.out.println(response);
        }
    }
}
