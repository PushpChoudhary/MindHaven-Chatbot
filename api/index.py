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
load_dotenv()
app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO)

# --- Global AI Components (Initialized as None) ---
# We define the variables here but will only load the heavy models on the first API call
llm = None
vector_db = None
qa_chain = None

# --- MongoDB Connection ---
try:
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_default_database() 
    
    appointments_collection = db.appointments
    chat_history_collection = db.chat_history
    
    logging.info("✅ Successfully connected to MongoDB.")
except Exception as e:
    logging.error(f"❌ Could not connect to MongoDB: {e}")
    db = None

# --- Health Check Route ---
@app.route('/health')
def health_check():
    return {"status": "ready"}, 200
# --- Add this new route to your index.py file ---

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "Welcome to the MindHaven Chatbot API!"
    })

# --- Your other routes like /book-appointment and /ask stay the same ---
# --- AI Initialization Function (Lazy Loading) ---
def initialize_ai_components():
    """
    This function loads the heavy AI models into the global variables.
    It's designed to run only once, on the first call to the /ask endpoint.
    """
    global llm, vector_db, qa_chain
    
    # The 'if qa_chain is None' check ensures this block runs only once.
    if qa_chain is None:
        logging.info("First request received. Initializing AI components...")
        try:
            llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="gemma2-9b-it")
            
            vector_db = Chroma(
                persist_directory="./chroma_db", 
                embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            )
            
            retriever = vector_db.as_retriever()
            prompt_template = "You are a friendly AI assistant...\n{context}\nUser: {question}\nChatbot:"
            PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever, 
                chain_type_kwargs={"prompt": PROMPT}
            )
            logging.info("✅ AI components initialized successfully.")
        except Exception as e:
            logging.error(f"❌ Error during AI component initialization: {e}")
            # Ensure qa_chain is not partially initialized
            qa_chain = None 

# --- API Routes ---
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
        appointment_doc = {
            "name": name, "email": email, "datetime_val": datetime_val, 
            "message": message, "submitted_at": datetime.now(timezone.utc)
        }
        appointments_collection.insert_one(appointment_doc)
        return jsonify({"message": f"Hi {name}, your appointment for {datetime_val} has been booked."})
    except Exception as e:
        logging.error(f"Error while booking appointment: {e}")
        return jsonify({"error": "Could not book appointment"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    # This line ensures the AI models are loaded before we use them.
    # After the first call, it will do nothing.
    initialize_ai_components()

    # If initialization failed, qa_chain will be None.
    if qa_chain is None:
        return jsonify({"error": "AI service is currently unavailable. Initialization failed."}), 503

    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default_session")

    if not user_message:
        return jsonify({"error": "Message is empty"}), 400
        
    try:
        chat_history_collection.insert_one({
            "session_id": session_id, "sender": 'user', 
            "message": user_message, "timestamp": datetime.now(timezone.utc)
        })
        
        # The 'qa_chain' variable is used here just like in your original code
        response = qa_chain.invoke({"query": user_message})
        response_text = response.get("result", "Sorry, there was an issue.")
        
        chat_history_collection.insert_one({
            "session_id": session_id, "sender": 'bot', 
            "message": response_text, "timestamp": datetime.now(timezone.utc)
        })
        
        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"Error in /ask route: {e}")
        return jsonify({"error": "Internal server error"}), 500

# This block is now only for local testing and is not needed for Vercel deployment.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"🚀 Server starting for local testing on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
