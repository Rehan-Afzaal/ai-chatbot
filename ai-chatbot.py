import logging
import os
import tempfile
import traceback
from threading import Thread
import openai
import pytesseract
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path


app = Flask(__name__, static_folder='static')
CORS(app)
app.debug = True

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Load environment variables
load_dotenv()
import certifi

ca = certifi.where()

# MongoDB setup
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri, tlsCAFile=ca)
db_name = os.getenv('DB_NAME')

file_collection = client.get_database(db_name).get_collection("chats")

# OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

#  Handles text extraction and chunking in a separate thread to avoid blocking the main application.
def background_processing(file_path, user_id, document_id):
    try:
        logging.info(f"Starting background processing for file: {file_path}")
        # Extract text from PDF File
        text = extract_text_from_pdf_chat(file_path)
        if text:
            # process the text into chunks and store into mongoDB
            chunks = process_text_to_chunks(text)
            logging.info(f"Chunks generated: {len(chunks)}")
            logging.info(f"Updating MongoDB with chunks for document ID: {document_id}")
            update_text_chunks(document_id, chunks, os.path.basename(file_path), user_id)
        else:
            logging.error(f"No text extracted from file: {file_path}")
        # Remove Temporary files
        os.remove(file_path)
        logging.info(f"Temporary file removed: {file_path}")
    except Exception as e:
        logging.error(f"Error in background processing: {e}")
        logging.error(traceback.format_exc())

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.info("Upload route accessed")
    # check if 'file' is in the request
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    # process the uploaded file
    uploaded_file = request.files['file']
    user_id = request.form.get('user_id', '')
    # check file type and save into temporary location
    file_type = uploaded_file.filename.split('.')[-1].lower()
    if file_type != 'pdf':
        logging.error("Unsupported file type")
        return jsonify({"error": "Unsupported file type"}), 400

    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    uploaded_file.save(temp_file)
    temp_file.close()

    # Create initial document entry and start background processing thread
    document_id = create_initial_document_entry(uploaded_file.filename, user_id)
    logging.info(f"Initial document entry created with ID: {document_id}")

    thread = Thread(target=background_processing, args=(temp_file.name, user_id, document_id))
    thread.start()

    logging.info("Started background processing thread")
    return jsonify({"message": "File upload received, processing started", "document_id": str(document_id)})


# Create initial document entry in MongoDB
def create_initial_document_entry(filename, user_id):
    initial_document = {"filename": filename, "content": [], "user_id": ObjectId(user_id)}
    result = file_collection.insert_one(initial_document)
    return result.inserted_id


# Update text chunks in MongoDB
def update_text_chunks(document_id, chunks, filename, user_id):
    try:
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError("Chunks must be a list of strings")
        logging.info(f"Performing MongoDB update for document ID: {document_id}")
        file_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"content": chunks, "filename": filename, "user_id": ObjectId(user_id)}}
        )
        logging.info(f"MongoDB update complete for document ID: {document_id}")
    except Exception as e:
        logging.error(f"Error in updating MongoDB: {e}")
        logging.error(traceback.format_exc())


# Route for handling user quries based on document content
@app.route('/chat', methods=['POST'])
def chat():
    # check if required Json data is available in the request
    if not request.json or 'question' not in request.json or 'document_id' not in request.json or 'user_id' not in request.json:
        return jsonify({'error': 'Missing data'}), 400

    # Extract relevant data from the request
    user_question = request.json['question']
    document_id = request.json['document_id']
    user_id = request.json['user_id']
    prompt = f"{system_prompt}{user_question}"
    # Retrieve document from MongoDB
    bson_document_id = ObjectId(document_id)
    document = file_collection.find_one({"_id": bson_document_id})
    if not document:
        return jsonify({"error": "Document not found"}), 404

    # Initiate chatbot and update MongoDB with user query and chatbot response
    # response = init_chatbot(document['content'], user_id)({'question': user_question},)(prompt)
    response = init_chatbot(document['content'], user_id, prompt)  

    file_collection.update_one({"_id": bson_document_id}, {"$push": {
        "chat_messages": {"user_id": user_id, "question": user_question, "response": response.get("answer")}}})
    return jsonify({"response": response.get("answer")})


system_prompt = """
Your task is to provide detailed, conversational, and factual responses to questions based on the context of specific document sections. Focus on the most relevant sections of the document to answer the query. Ensure your responses are thorough, accurate, and maintain a conversational tone, reflecting the in-depth knowledge contained in the document. Be informative and relevant to the user's query.

Question: {user_question}
Relevant Section: {section_reference}
Based on the above guidelines and the document's content, please provide a detailed response to the question.

"""

# Route for checking server status
@app.route("/", methods=["GET"])
def index():
    return "Server is running.."


# Extract text from a PDF file using Tesseract OCR
def extract_text_from_pdf_chat(file_path):
    try:
        # Implementation using pytesseract and pdf2image
        logging.info(f"Extracting text from PDF: {file_path}")
        images = convert_from_path(file_path)
        text = ''.join(pytesseract.image_to_string(image) for image in images)
        logging.info(f"Extracted text length: {len(text)} characters")
        return text
    except Exception as e:
        logging.error(f"Error in extracting text from PDF: {e}")
        logging.error(traceback.format_exc())
        return ""

# Creation Chunks using NLP
def process_text_to_chunks(text):
    # Use NLP library for semantic splitting (e.g., spaCy)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = [str(span) for span in doc.sents]  # Splitting by sentences as an example
    return chunks

# Store text chunks in MongoDB
def store_text_chunks(chunks, filename, user_id):
    document = {"filename": filename, "content": chunks, "user_id": ObjectId(user_id)}
    result = file_collection.insert_one(document)
    return result.inserted_id


def init_chatbot(text_chunks, user_id, prompt):
    # Retrieve chat history from MongoDB, handling cases where no history exists
    chat_history = file_collection.find_one({"_id": ObjectId(user_id)})
    past_messages = chat_history["chat_messages"] if chat_history else []

    # Create text embeddings for efficient knowledge base search
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Build a knowledge base using FAISS for fast retrieval
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    # Initialize the GPT model for generating responses
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4-1106-preview", temperature=0.1)

    # Establish conversational memory to store past messages
    memory = ConversationBufferMemory(memory_key="chat_history", initial_messages=past_messages)
    # Construct the retrieval chain, combining knowledge base, GPT model, and memory
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=knowledge_base.as_retriever(), memory=memory)
    response = chain(prompt)
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1338)