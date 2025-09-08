
import logging
import os
from datetime import datetime, timezone # timezone ko import kiya
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient

# LangChain aur AI se related imports waise hi rahenge
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Setup ---
# .env file se saari keys load karega
load_dotenv()
app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO)

# --- MongoDB Connection ---
# Yahan hum SQLAlchemy ko hata kar PyMongo ka istemaal kar rahe hain
try:
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    # Aapke MONGO_URI se database ka naam ('MindHavenDB') automatically le lega
    db = client.get_default_database() 
    
    # "Collections" banayenge (yeh SQL mein tables jaise hote hain)
    appointments_collection = db.appointments
    chat_history_collection = db.chat_history
    
    logging.info("✅ MongoDB se connection safaltapoorvak jud gaya hai.")
except Exception as e:
    logging.error(f"❌ MongoDB se connect nahi ho paaya: {e}")
    db = None # Agar connection fail hota hai toh db ko None set kar denge

# --- Health Check Route (Koi Change Nahi) ---
@app.route('/health')
def health_check():
    return {"status": "ready"}, 200

# --- AI Functions (Inmein Koi Change Nahi Hoga) ---
def initialize_llm():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.error("GROQ_API_KEY environment variable nahi mila.")
            return None
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="gemma2-9b-it")
        logging.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logging.error(f"LLM initialize karte waqt error: {e}")
        return None

def setup_qa_chain(vector_db, llm):
    # Yeh function ChromaDB (AI ki library) ke liye hai, iska MongoDB se lena-dena nahi hai
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
        logging.error(f"QA chain set up karte waqt error: {e}")
        return None

# --- Routes jinko MongoDB ke liye Update Kiya Gaya Hai ---
@app.route('/book-appointment', methods=['POST'])
def book_appointment():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    datetime_val = data.get('datetime')
    message = data.get('message', '')

    if not all([name, email, datetime_val]):
        return jsonify({"error": "Name, email, and datetime zaroori hain"}), 400
    
    try:
        # Ek "document" (dictionary) banayenge
        appointment_doc = {
            "name": name,
            "email": email,
            "datetime_val": datetime_val,
            "message": message,
            "submitted_at": datetime.now(timezone.utc) # Updated this line
        }
        # Is document ko 'appointments' collection mein daal denge
        appointments_collection.insert_one(appointment_doc)
        return jsonify({"message": f"Hi {name}, aapki appointment {datetime_val} ke liye book ho gayi hai."})
    except Exception as e:
        logging.error(f"Appointment book karte waqt error: {e}")
        return jsonify({"error": "Appointment book nahi ho paayi"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default_session")

    if not user_message:
        return jsonify({"error": "Message khaali hai"}), 400
        
    try:
        # User ke message ko MongoDB mein save karenge
        chat_history_collection.insert_one({
            "session_id": session_id,
            "sender": 'user',
            "message": user_message,
            "timestamp": datetime.now(timezone.utc) # Updated this line
        })
        
        # AI se jawaab lenge
        response = qa_chain.invoke({"query": user_message})
        response_text = response.get("result", "Maaf kijiye, kuch samasya aa gayi.")
        
        # Bot ke jawaab ko MongoDB mein save karenge
        chat_history_collection.insert_one({
            "session_id": session_id,
            "sender": 'bot',
            "message": response_text,
            "timestamp": datetime.now(timezone.utc) # And this line
        })
        
        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"/ask route mein error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# --- Main Execution Block ---
if __name__ == "__main__":
    if db is None:
        logging.critical("Database connection fail ho gaya. Application band ho rahi hai.")
    else:
        llm = initialize_llm()
        # ChromaDB waise hi load hoga, iska MongoDB se koi connection nahi hai
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
        qa_chain = setup_qa_chain(vector_db, llm)
        
        if not all([llm, vector_db, qa_chain]):
            logging.critical("AI ka koi component fail ho gaya. Application band ho rahi hai.")
        else:
            logging.info("🚀 Server shuru ho raha hai http://0.0.0.0:5000 par")
            app.run(host='0.0.0.0', port=5000, debug=False)

