import logging
import os
import csv
from datetime import datetime
import sqlite3

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# SQLite DB initialization for chat history
def init_db():
    with sqlite3.connect("chat_history.db") as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                sender TEXT,
                message TEXT
            )
        ''')

# CSV file for appointment booking
CSV_FILE = 'appointments.csv'
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Email', 'DateTime', 'Message', 'SubmittedAt'])

# Appointment booking endpoint
@app.route('/book-appointment', methods=['POST'])
def book_appointment():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    datetime_val = data.get('datetime')
    message = data.get('message', '')

    if not name or not email or not datetime_val:
        return jsonify({"error": "Name, email, and datetime are required"}), 400

    try:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, email, datetime_val, message, datetime.now().isoformat()])
        logging.info(f"Appointment saved: {name}, {email}, {datetime_val}")
        return jsonify({"message": f"Hi {name}, your appointment on {datetime_val} has been booked. We'll reach out to {email}."})
    except Exception as e:
        logging.error(f"Failed to save appointment: {e}", exc_info=True)
        return jsonify({"error": "Failed to book appointment"}), 500

import os
from dotenv import load_dotenv

load_dotenv()

def initialize_llm():
    try:
        # Get the API key from the environment variable
        groq_api_key = os.getenv("GROQ_API_KEY")
        print(f"--- Loaded API Key: {groq_api_key} ---") 
        if not groq_api_key:
            logging.error("GROQ_API_KEY environment variable not set.")
            return None

            

        llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,  # Use the variable loaded from .env
            model_name="gemma2-9b-it" # Or whichever model you want to use
        )
        logging.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}", exc_info=True)
        return None

# Create or load vector DB from PDFs
def create_vector_db():
    try:
        loader = DirectoryLoader("./Data/", glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            logging.warning("No documents found in ./Data/")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
        vector_db.persist()
        logging.info("ChromaDB created and persisted")
        return vector_db
    except Exception as e:
        logging.error(f"Error creating vector DB: {e}", exc_info=True)
        return None

# Setup QA chain with prompt template
def setup_qa_chain(vector_db, llm):
    try:
        retriever = vector_db.as_retriever()

        prompt_template = """
You are a friendly, empathetic, and supportive mental health chatbot. Your role is to act like a genuine friend who listens carefully and responds with kindness, warmth, and understanding. Avoid sounding robotic or overly formal. Always focus on the user's emotional well-being and guide them thoughtfully through their concerns.

When the user is casually chatting or greeting you (like saying "hi" or "how are you"), respond in a light, friendly tone — do **not** mention anything about stored data or context.

However, when the user expresses a concern, problem, or question related to their mental or emotional health, provide thoughtful, practical, and supportive responses based on the context. Use your knowledge and compassion to help them feel heard and supported. Never say you were given the context or data — just respond naturally as if you're simply a good friend who's always been listening.

Conversation context:
{context}

User: {question}
Chatbot:
"""

        # Change input variable 'query' -> 'question' to match chain input key
        PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        logging.info("QA chain set up successfully")
        return qa_chain
    except Exception as e:
        logging.error(f"Error setting up QA chain: {e}", exc_info=True)
        return None

# Get conversation history text for context
def get_conversation_history(session_id):
    with sqlite3.connect("chat_history.db") as conn:
        cursor = conn.execute(
            "SELECT sender, message FROM chat WHERE session_id = ? ORDER BY id ASC", (session_id,)
        )
        rows = cursor.fetchall()

    history_text = ""
    for sender, message in rows:
        # Format with capitalized sender to align with prompt style
        sender_label = "User" if sender.lower() == "user" else "Chatbot"
        history_text += f"{sender_label}: {message}\n"
    return history_text

# Core chat function
def chat(session_id, query):
    global qa_chain, llm, vector_db
    try:
        logging.info(f"chat() called with session_id={session_id}, query={query}")

        if not query or query.strip() == "":
            return {"error": "No message provided or query is empty"}

        if qa_chain is None or llm is None or vector_db is None:
            return {"error": "QA chain, LLM, or vector database not initialized properly"}

        conversation_context = get_conversation_history(session_id)
        logging.info(f"Conversation context retrieved: {conversation_context}")

        response_text = qa_chain.invoke({
            "query": query,
            "context": conversation_context
        })

        logging.info(f"QA chain response: {response_text}")

        if not response_text.get("result", "").strip():
            return {"error": "Received empty response from QA chain"}

        return {"response": response_text["result"]}

    except Exception as e:
        logging.error(f"Error in chat function: {e}", exc_info=True)
        return {"error": f"Internal server error: {str(e)}"}

# Chat endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default_session")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        with sqlite3.connect("chat_history.db") as conn:
            conn.execute(
                "INSERT INTO chat (session_id, sender, message) VALUES (?, ?, ?)",
                (session_id, 'user', user_message)
            )
            conn.commit()

        bot_response = chat(session_id, user_message)

        if "error" in bot_response:
            logging.error(f"Chatbot error: {bot_response['error']}")
            return jsonify(bot_response), 500

        response_text = bot_response.get('response', 'Error')

        with sqlite3.connect("chat_history.db") as conn:
            conn.execute(
                "INSERT INTO chat (session_id, sender, message) VALUES (?, ?, ?)",
                (session_id, 'bot', response_text)
            )
            conn.commit()

        return jsonify({"response": response_text})

    except Exception as e:
        logging.error(f"Exception in /ask endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# Health check endpoint
@app.route("/health")
def health_check():
    status = {
        "llm_initialized": llm is not None,
        "vector_db_initialized": vector_db is not None,
        "qa_chain_initialized": qa_chain is not None,
    }
    all_ready = all(status.values())
    return jsonify({
        "status": "ready" if all_ready else "not ready",
        "details": status
    }), 200 if all_ready else 503


if __name__ == "__main__":
    init_db()

    llm = initialize_llm()
    db_path = "./chroma_db"
    vector_db = None

    if not os.path.exists(db_path):
        logging.info("Creating vector DB...")
        vector_db = create_vector_db()
        if vector_db is None:
            logging.error("Failed to create vector database.")
    else:
        try:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
            logging.info("Loaded existing Chroma vector DB")
        except Exception as e:
            logging.error(f"Failed to load existing vector DB: {e}", exc_info=True)

    qa_chain = None
    if vector_db and llm:
        qa_chain = setup_qa_chain(vector_db, llm)

    app.run(host='0.0.0.0', port=5000, debug=True)
