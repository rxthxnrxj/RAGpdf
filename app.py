from flask import Flask, render_template, request, jsonify
# ... (keep your existing imports) ...
from dotenv import load_dotenv
import os
import requests
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client
from langchain_openai.embeddings import OpenAIEmbeddings
from ollama import Client as OllamaClient
from pydantic_settings import BaseSettings
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings

import glob
app = Flask(__name__)

# Move your existing functions here...


def process_pdfs_in_folder(folder_path, collection):
    """Processes all PDFs in a folder and adds their content to ChromaDB."""
    try:
        # Get all PDF files in the folder
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in the folder: {folder_path}")
            return

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path}")
            process_pdf_and_add_to_chromadb(pdf_path, collection)
            print(f"Added '{pdf_path}' to the knowledge base.")
    except Exception as e:
        print(f"Error processing PDFs in folder {folder_path}: {e}")


# Step 1: PDF Extraction and Text Splitting


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF {pdf_path}: {e}")


def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    """Splits text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Step 2: Initialize ChromaDB


def initialize_chromadb():
    """Initializes the ChromaDB client."""
    client = Client()
    collection = client.get_or_create_collection("rag_content")
    return collection


def add_content_to_chromadb(content, source_id, collection):
    """Adds content to ChromaDB after embedding."""
    chunks = split_text_into_chunks(content)
    for i, chunk in enumerate(chunks):
        try:
            # Generate embeddings for the chunk
            embedding = embedding_model.embed_documents([chunk])
            collection.add(
                documents=[chunk],
                metadatas=[{"source": f"{source_id}_chunk_{i}"}],
                ids=[f"{source_id}_chunk_{i}"]
            )
        except Exception as e:
            print(f"Error adding chunk {i} to ChromaDB: {e}")


def process_pdf_and_add_to_chromadb(pdf_path, collection):
    """Processes a PDF and adds its content to ChromaDB."""
    try:
        content = extract_text_from_pdf(pdf_path)
        add_content_to_chromadb(content, pdf_path, collection)
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

# Step 4: Document Retrieval


def retrieve_documents(query, collection):
    """Retrieve relevant documents from ChromaDB."""
    try:
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        return results["documents"]
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


# Step 5: Initialize Ollama Client and Generate Response


def generate_response(query, context):
    """Generate a response using the local Ollama server."""
    try:
        # Define the prompt with context and query
        prompt = f"""You are an intelligent assistant. You should always use only the provided context to answer the query.
                    If the relevant answer is not found in the context or if the context is missing or empty, ONLY respond with "Not found."

                    Context:
                    {context}

                    Query:
                    {query}

                    Answer:
                    """

        # API endpoint of the local Ollama server
        url = "http://127.0.0.1:11434/api/generate"

        # Payload for the POST request
        payload = {
            "model": "mistral",  # Replace with the valid model available on your server
            "prompt": prompt
        }

        # Send the request to the local server
        response = requests.post(url, json=payload)

        # Check the response status
        if response.status_code == 200:
            return response.json().get("text", "No response text found.")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error generating response: {e}. Please ensure the server is running and the model name is correct."


def generate_response_claude(query, context):
    """Generate a response using Claude API."""
    try:
        # Define the prompt with context and query
        messages = [
            {
                "role": "user",
                "content": f"""You are an intelligent assistant. You should only use the provided context to answer the query .
                    If the relevant answer is not found in the context or if the context is missing or empty, ONLY respond with "Not found." 

                    Context:
                    {context}

                    Query:
                    {query}

                    Answer:
                    """
            }
        ]

        # API endpoint for Claude
        url = "https://api.anthropic.com/v1/messages"

        # Headers with API key
        headers = {
            # Recommended: use environment variable
            "x-api-key": "",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"  # Update to the current API version
        }

        # Payload for the API request
        payload = {
            "model": "claude-3-haiku-20240307",  # Updated model name
            "messages": messages,
            "max_tokens": 300
        }

        # Send the POST request
        response = requests.post(url, json=payload, headers=headers)

        # Check the response status
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error generating response: {e}. Please ensure the API key and configuration are correct."


def rag_pipeline(query, collection):
    """Complete RAG pipeline: retrieve context and generate response."""
    retrieved_docs = retrieve_documents(query, collection)
    if not retrieved_docs:
        return "No relevant documents found."

    # Flatten nested lists in retrieved_docs
    flattened_docs = [doc for sublist in retrieved_docs for doc in sublist]

    # Join all documents into a single string
    context = "\n".join(flattened_docs)

    print("\n\nContext for the query:", context)
    return generate_response_claude(query, context)


def interactive_cli():
    """Runs an interactive CLI for the application."""
    print("\n\nWelcome to the RAG-based Interactive System!")
    try:
        collection = initialize_chromadb()
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return

    while True:
        print("\nMenu:")
        print("1. Add a single PDF to the knowledge base")
        print("2. Add multiple PDFs from a folder")
        print("3. Ask a question")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            pdf_path = input("Enter the path to the PDF file: ")
            process_pdf_and_add_to_chromadb(pdf_path, collection)
            print(f"PDF '{pdf_path}' added to the knowledge base.")

        elif choice == "2":
            folder_path = input("Enter the folder path containing PDF files: ")
            process_pdfs_in_folder(folder_path, collection)

        elif choice == "3":
            query = input("Enter your question: ")
            try:
                response = rag_pipeline(query, collection)
                print(f"\n\nResponse:\n{response}")
            except Exception as e:
                print(f"Error retrieving answer: {e}")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and file.filename.endswith('.pdf'):
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the file temporarily and process it
            collection = initialize_chromadb()
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)

            try:
                # Process the PDF and create embeddings
                process_pdf_and_add_to_chromadb(file_path, collection)
            finally:
                # Clean up temp file even if processing fails
                if os.path.exists(file_path):
                    os.remove(file_path)

            return jsonify({
                'message': 'File processed successfully',
                'status': 'complete'
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload a PDF file.'})

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')

    collection = initialize_chromadb()
    response = rag_pipeline(query_text, collection)

    # Format the response if it contains code
    formatted_response = response
    if '```' in response:
        # The response already contains markdown-style code blocks
        formatted_response = response
    else:
        # If there's no code block formatting but it looks like code
        if any(keyword in response for keyword in ['def ', 'class ', 'import ', '//']):
            formatted_response = f"```python\n{response}\n```"

    return jsonify({'response': formatted_response})


if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    app.run(debug=True)
