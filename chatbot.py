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
load_dotenv()
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///local_test.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Models ---
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

# --- Your Core AI Functions (Unchanged) ---
def initialize_llm():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.error("GROQ_API_KEY environment variable not set.")
            return None
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="gemma2-9b-it")
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
        prompt_template = "You are a friendly AI assistant...\n{context}\nUser: {question}\nChatbot:"
        PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
        )
        logging.info("QA chain set up successfully")
        return qa_chain
    except Exception as e:
        logging.error(f"Error setting up QA chain: {e}", exc_info=True)
        return None

# --- Updated Routes ---
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
        return jsonify({"message": f"Hi {name}, your appointment on {datetime_val} has been booked."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to book appointment"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_message, session_id = data.get("message", ""), data.get("session_id", "default_session")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        db.session.add(ChatHistory(session_id=session_id, sender='user', message=user_message))
        db.session.commit()
        response = qa_chain.invoke({"query": user_message})
        response_text = response.get("result", "Sorry, I encountered an issue.")
        db.session.add(ChatHistory(session_id=session_id, sender='bot', message=response_text))
        db.session.commit()
        return jsonify({"response": response_text})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

# --- Main Execution Block ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    llm = initialize_llm()
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
    qa_chain = setup_qa_chain(vector_db, llm)
    app.run(host='0.0.0.0', port=5000, debug=False)