import logging
import os
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient

# LangChain and AI-related imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Setup ---
# Load all keys from the .env file
load_dotenv()
app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO)

# --- MongoDB Connection ---
try:
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    # The database name will be taken automatically from your MONGO_URI
    db = client.get_default_database() 
    
    # Create "Collections" (similar to tables in SQL)
    appointments_collection = db.appointments
    chat_history_collection = db.chat_history
    
    logging.info("✅ Successfully connected to MongoDB.")
except Exception as e:
    logging.error(f"❌ Could not connect to MongoDB: {e}")
    db = None # Set db to None if connection fails

# --- Health Check Route ---
@app.route('/health')
def health_check():
    return {"status": "ready"}, 200

# --- AI Functions ---
def initialize_llm():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.error("GROQ_API_KEY environment variable not found.")
            return None
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="gemma2-9b-it")
        logging.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logging.error(f"Error while initializing LLM: {e}")
        return None

def setup_qa_chain(vector_db, llm):
    # This function is for ChromaDB, it is independent of MongoDB
    try:
        retriever = vector_db.as_retriever()
        prompt_template = "You are a friendly AI assistant...\n{context}\nUser: {question}\nChatbot:"
        PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
        )
        logging.info("QA chain set up successfully")
        return qa_chain
    except Exception as e:
        logging.error(f"Error while setting up QA chain: {e}")
        return None

# --- API Routes Updated for MongoDB ---
@app.route('/book-appointment', methods=['POST'])
def book_appointment():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    datetime_val = data.get('datetime')
    message = data.get('message', '')

    if not all([name, email, datetime_val]):
        return jsonify({"error": "Name, email, and datetime are required"}), 400
    
    try:
        # Create a document (dictionary) to insert
        appointment_doc = {
            "name": name,
            "email": email,
            "datetime_val": datetime_val,
            "message": message,
            "submitted_at": datetime.now(timezone.utc)
        }
        # Insert the document into the 'appointments' collection
        appointments_collection.insert_one(appointment_doc)
        return jsonify({"message": f"Hi {name}, your appointment for {datetime_val} has been booked."})
    except Exception as e:
        logging.error(f"Error while booking appointment: {e}")
        return jsonify({"error": "Could not book appointment"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default_session")

    if not user_message:
        return jsonify({"error": "Message is empty"}), 400
        
    try:
        # Save user's message to MongoDB
        chat_history_collection.insert_one({
            "session_id": session_id,
            "sender": 'user',
            "message": user_message,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Get response from AI
        response = qa_chain.invoke({"query": user_message})
        response_text = response.get("result", "Sorry, there was an issue.")
        
        # Save bot's response to MongoDB
        chat_history_collection.insert_one({
            "session_id": session_id,
            "sender": 'bot',
            "message": response_text,
            "timestamp": datetime.now(timezone.utc)
        })
        
        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"Error in /ask route: {e}")
        return jsonify({"error": "Internal server error"}), 500

# --- Main Execution Block ---
# This block runs ONLY when you execute `python index.py` on your local machine.
# On Render, Gunicorn runs the `app` object directly and this block is ignored.
if __name__ == "__main__":
    if db is None:
        logging.critical("Database connection failed. Application is shutting down.")
    else:
        llm = initialize_llm()
        # ChromaDB is loaded here, it has no dependency on the MongoDB connection
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
        qa_chain = setup_qa_chain(vector_db, llm)
        
        if not all([llm, vector_db, qa_chain]):
            logging.critical("An AI component failed to initialize. Application is shutting down.")
        else:
            # Get the port from the environment variable for Render, default to 5000 for local dev
            port = int(os.environ.get('PORT', 5000))
            logging.info(f"🚀 Server starting on http://0.0.0.0:{port}")
            app.run(host='0.0.0.0', port=port, debug=False)
