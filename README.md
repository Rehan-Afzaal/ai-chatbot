
# PDF Chatbot Application

## Description
This application allows users to upload PDF documents, extracts text from them, and provides a conversational interface where users can ask questions about the document's content. It uses OCR for text extraction, MongoDB for data storage, and an AI-powered chatbot for answering queries.

## Features
- **PDF Upload**: Users can upload PDF documents.
- **Text Extraction**: Extracts text from PDF using OCR.
- **Text Chunking**: Processes extracted text into manageable chunks.
- **Question Answering**: Users can ask questions about the document, answered by an AI chatbot.
- **Database Storage**: Uses MongoDB to store document data and chat history.

## How to Run
1. Set environment variables for MongoDB and OpenAI API key.
2. Start the Flask server with `python [filename].py`.
3. Access the application through the specified host and port.

## Dependencies
- Flask
- Pytesseract
- MongoDB
- OpenAI GPT
- pdf2image
- dotenv
- Spacy (for NLP tasks)

## Additional Notes
- Ensure you have MongoDB running and accessible.
- The OpenAI API key needs to be set in the environment variables for the chatbot functionality.
