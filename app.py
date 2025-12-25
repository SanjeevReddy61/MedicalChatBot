from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from langdetect import detect
import os

# -------------------- PROJECT IMPORTS --------------------
from src.helper import download_embeddings
from src.prompt import SYSTEM_PROMPT

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------- APP SETUP --------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing")

# -------------------- EMBEDDINGS & VECTOR STORE --------------------
embeddings = download_embeddings()

index_name = "medical-chatbot"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -------------------- LLM (GROQ – STABLE & FAST) --------------------
chat_model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3
)


# -------------------- PROMPT --------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
    ]
)

# -------------------- RAG CHAIN (MODERN LANGCHAIN) --------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | chat_model
    | StrOutputParser()
)

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/detect_language", methods=["POST"])
def detect_language():
    text = request.form.get("text", "")
    lang = detect(text)

    lang_map = {
        "en": "en-US",
        "hi": "hi-IN",
        "te": "te-IN",
        "fr": "fr-FR",
        "es": "es-ES",
        "de": "de-DE"
    }

    return jsonify({"lang": lang_map.get(lang, "en-US")})

def is_greeting(text: str) -> bool:
    greetings = [
        "hi", "hello", "hey", "hii",
        "how are you", "how r you",
        "good morning", "good afternoon", "good evening",
        "what's up", "whats up"
    ]
    text = text.lower().strip()
    return any(greet in text for greet in greetings)

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]

    # 🟢 Handle greetings naturally (NO RAG)
    if is_greeting(user_message):
        return (
            "Hello 😊 I’m doing well, thanks for asking! "
            "I can help you with medical-related questions. "
            "What would you like to know?"
        )

    # 🟢 Use RAG for actual medical questions
    try:
        answer = rag_chain.invoke(user_message)
        return answer

    except Exception as e:
        print("LLM Error:", e)
        return (
            "⚠️ The medical assistant is temporarily unavailable. "
            "Please consult a licensed doctor."
        ), 503


# -------------------- RUN SERVER --------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8080,
        debug=False,
        use_reloader=False
    )
