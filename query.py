import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load API key
load_dotenv()

CHROMA_PATH = "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Template for our RAG prompt
RAG_PROMPT_TEMPLATE = """
Using ONLY the context below, answer the question.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.

Context:
{context}

Question:
{question}
"""

def main():
    # --- Setup ---
    # Create a command-line parser to accept a question
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("query_text", type=str, help="The question to ask the document.")
    args = parser.parse_args()
    query_text = args.query_text

    # 1. Initialize the embedding model
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # 2. Load the existing ChromaDB
    print(f"Loading vector database from {CHROMA_PATH}...")
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_function
    )
    print("Database loaded.")

    # 3. Get the retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})  # "k": 3 means get top 3 relevant chunks

    # 4. Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 5. Create the RAG prompt template
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 6. Create the RAG chain
    print("Setting up RAG chain...")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("Ready to answer questions.")

    # --- Ask the Question ---
    print(f"\nQuerying for: '{query_text}'")
    
    # Invoke the chain and get the response
    response = rag_chain.invoke(query_text)
    
    print("\n--- Answer ---")
    print(response)


if __name__ == "__main__":
    main()