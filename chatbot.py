import logging
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Setup ---
load_dotenv()  # Loads variables from .env file
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- Database Configuration ---
# This will get the DATABASE_URL we set on Render from environment variables
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///chat_history.db') # Uses local sqlite if DATABASE_URL is not set
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Models (Replaces CSV and old DB) ---
class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    datetime_val = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=True)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    sender = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)

# --- Your Core AI and LangChain Functions (UNCHANGED) ---
def initialize_llm():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.error("GROQ_API_KEY environment variable not set.")
            return None
        llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it" # Using the latest working model
        )
        logging.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}", exc_info=True)
        return None

def create_vector_db():
    try:
        loader = DirectoryLoader("./Data/", glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            logging.warning("No documents found in ./Data/")
            return None
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

def setup_qa_chain(vector_db, llm):
    try:
        retriever = vector_db.as_retriever()
        prompt_template = """
        You are a friendly, empathetic, and supportive mental health chatbot...
        {context}
        User: {question}
        Chatbot:
        """
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

# --- Updated Flask Routes (Using the New Database) ---
@app.route('/book-appointment', methods=['POST'])
def book_appointment():
    data = request.get_json()
    name, email, datetime_val = data.get('name'), data.get('email'), data.get('datetime')
    message = data.get('message', '')

    if not all([name, email, datetime_val]):
        return jsonify({"error": "Name, email, and datetime are required"}), 400

    try:
        new_appointment = Appointment(name=name, email=email, datetime_val=datetime_val, message=message)
        db.session.add(new_appointment)
        db.session.commit()
        logging.info(f"Appointment saved for {name}")
        return jsonify({"message": f"Hi {name}, your appointment on {datetime_val} has been booked."})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Failed to save appointment: {e}", exc_info=True)
        return jsonify({"error": "Failed to book appointment"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default_session")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Save user message to DB
        user_chat = ChatHistory(session_id=session_id, sender='user', message=user_message)
        db.session.add(user_chat)
        db.session.commit()

        # Get conversation history for context
        history = ChatHistory.query.filter_by(session_id=session_id).order_by(ChatHistory.id.asc()).all()
        conversation_context = "\n".join([f"{'User' if entry.sender == 'user' else 'Chatbot'}: {entry.message}" for entry in history])
        
        # Get response from LLM
        response = qa_chain.invoke({"query": user_message})
        response_text = response.get("result", "Sorry, I encountered an issue.")

        # Save bot response to DB
        bot_chat = ChatHistory(session_id=session_id, sender='bot', message=response_text)
        db.session.add(bot_chat)
        db.session.commit()

        return jsonify({"response": response_text})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Exception in /ask endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health")
def health_check():
    return jsonify({"status": "ready"}), 200

# --- Main Execution Block ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # This creates your tables if they don't exist

    llm = initialize_llm()
    db_path = "./chroma_db"
    vector_db = None

    # This logic for loading/creating chroma_db remains the same
    if not os.path.exists(db_path):
        logging.info("Creating vector DB...")
        vector_db = create_vector_db()
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