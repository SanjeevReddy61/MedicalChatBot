from flask import Flask, render_template,jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from langdetect import detect

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

@app.route("/detect_language", methods=["POST"])
def detect_language():
    text = request.form.get("text", "")
    lang = detect(text)  # auto-detect language
    # Map to a voice code for SpeechSynthesis
    lang_map = {
        "en": "en-US",
        "hi": "hi-IN",
        "te": "te-IN",
        "fr": "fr-FR",
        "es": "es-ES",
        "de": "de-DE"
    }
    return jsonify({"lang": lang_map.get(lang, "en-US")})

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_lang = detect(msg)   # detect language

    # Add dynamic instruction for Gemini
    lang_instruction = f"Always respond in the same language as the user (detected: {input_lang})."

    # Build prompt with language control
    custom_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + " " + lang_instruction),
            ("human", "{input}"),
        ]
    )

    # Create a fresh chain using the custom prompt
    question_answer_chain = create_stuff_documents_chain(chatModel, custom_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])

    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
