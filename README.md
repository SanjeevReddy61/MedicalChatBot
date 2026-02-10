# Medical Chatbot using Retrieval-Augmented Generation (RAG)

MedicalChatBot

#📌 Overview

This project implements an AI-powered Medical Chatbot using Retrieval-Augmented Generation (RAG) to provide accurate, context-aware medical information.
Instead of relying solely on a language model, the chatbot retrieves relevant information from a trusted medical textbook (Encyclopedia of Medicine) and generates responses grounded in that knowledge, reducing hallucinations and improving reliability.

🎯 Objective

Build a reliable medical chatbot that provides factual, context-grounded answers

Minimize hallucinations commonly seen in LLM-only systems

Demonstrate an end-to-end GenAI system including retrieval, generation, and deployment

Pinecone vector database for storing and searching embeddings

Flask backend for running the chatbot locally

🧠 System Architecture

User submits a medical query via UI

Query is embedded using Hugging Face embeddings

Similar medical text chunks are retrieved from Pinecone

Retrieved context is passed to the LLM via Groq API

LLM generates a grounded, meaningful response

Flask backend returns the response to the UI

📚 Dataset

Source: Encyclopedia of Medicine

Size: ~700 pages of medical text

Content: Diseases, symptoms, causes, and treatments


🛠️ Tech Stack

Programming Language: Python

Framework: Flask

LLM Inference: Groq API

Embeddings: Hugging Face Embedding Models

Vector Database: Pinecone

RAG Framework: LangChain

⚙️ Implementation Details
🔹 Text Processing

Medical text is cleaned and segmented using RecursiveCharacterTextSplitter

Chunk size: ~500 tokens

Chunk overlap: 50 tokens

Ensures preservation of medical context

🔹 Embeddings & Vector Store

Each chunk is converted into dense vector embeddings

Total vectors stored: ~6,000+

Stored and indexed in Pinecone for fast similarity search

🔹 Retrieval-Augmented Generation

Top-k (k = 3–5) relevant chunks retrieved per query

Retrieved context is injected into the LLM prompt

Reduces hallucinations and improves factual accuracy

🔹 Backend

Flask REST API handles:

User queries

Retrieval pipeline

LLM response generation

Enables real-time interaction

🔐 Safety Considerations

Responses are informational only

Includes medical disclaimers

Chatbot does not replace professional medical advice

Users are encouraged to consult healthcare professionals

🚀 Features

Context-aware medical responses

Reduced hallucinations using RAG

Scalable vector search with Pinecone

Low-latency inference using Groq

Modular and extensible architecture

🧪 Evaluation

Tested with diverse medical queries

Verified relevance and grounding of responses

Manual validation of response accuracy against source content

▶️ How to Run the Project

  1️⃣ Clone the Repository
  
     https://github.com/SanjeevReddy61/MedicalChatBot
  
  2️⃣ Install Dependencies 
  
     pip install -r requirements.txt
  3️⃣ Set Environment Variables
     
     export PINECONE_API_KEY=your_key
     export GROQ_API_KEY=your_key
  4️⃣Run the Flask App
  
     python app.py
