import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration

DATA_PATH = "data/"
PDF_NAME = "The_Invisible_Man.pdf" 
CHROMA_PATH = "chroma_db"

def load_document():
    """
    Loads the specified PDF document from the data path.
    """
    pdf_path = os.path.join(DATA_PATH, PDF_NAME)
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return None

    print(f"Loading document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")
    return pages

def split_documents(documents):
    """
    Splits the loaded documents into smaller chunks.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")
    return chunks

def create_and_store_embeddings(chunks):
    """
    Creates embeddings for the chunks and stores them in ChromaDB.
    """
    print("Creating embeddings and storing in ChromaDB (using local model)...")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Initialize the local embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs=model_kwargs 
    )
    # ----------------------------

    # Create the ChromaDB vector store
    db = Chroma.from_documents(
        chunks, 
        embedding_model, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Successfully created and saved vector store at {CHROMA_PATH}")

# --- Main execution ---
if __name__ == "__main__":
    # 1. Load the document
    documents = load_document()
    
    if documents:
        # 2. Split the document into chunks
        chunks = split_documents(documents)
        
        # 3. Create embeddings and store in ChromaDB
        create_and_store_embeddings(chunks)