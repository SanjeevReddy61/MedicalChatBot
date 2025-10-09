# MedicalChatBot

MedicalChatBot

This is a medical chatbot built using LangChain, Pinecone, Gemini API, and Flask. It uses a retrieval-augmented generation (RAG) pipeline to answer medical queries based on curated data.

Features

Retrieves and generates answers for medical-related questions

Pinecone vector database for storing and searching embeddings

Flask backend for running the chatbot locally

Setup

Clone the repo:

git clone https://github.com/SanjeevReddy61/MedicalChatBot.git
cd MedicalChatBot


Install requirements:

pip install -r requirements.txt


Run the app:

python app.py


Open in browser:

http://127.0.0.1:5000/

Project Structure

app.py → main Flask app

store_index.py → creates and stores embeddings in Pinecone

templates/ → HTML templates

static/ → static files

requirements.txt → dependencies