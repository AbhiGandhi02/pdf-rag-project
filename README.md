# PDF-RAG: "Chat with Your PDF"

This project is a simple, local-first Retrieval-Augmented Generation (RAG) application. It allows you to "chat" with a PDF document by loading its content into a vector database and using a Large Language Model (LLM) to answer questions based *only* on the document's content.

This project is built with LangChain and uses a local model for embeddings (free) and is configured to use OpenAI for answer generation (requires an API key).

## 🚀 Features

* **PDF Loading:** Loads text content from any PDF file.
* **Local Embeddings:** Uses a free, high-quality Hugging Face model (`all-MiniLM-L6-v2`) to generate vector embeddings, so this step costs nothing.
* **Local Vector Store:** Uses `ChromaDB` to store all vectors and text chunks locally in a `chroma_db` folder.
* **RAG Pipeline:** Ensures that the LLM's answers are sourced *from the document's text*, reducing hallucinations and allowing it to answer questions about private data.

---

## ⚙️ How It Works

The project is split into two main scripts:

### 1. Ingestion (`ingest.py`)

This script is responsible for "learning" the document.
1.  **Loads** the PDF from the `/data` folder using `PyPDFLoader`.
2.  **Splits** the document's text into small, overlapping chunks.
3.  **Embeds** each chunk by running it through the local `all-MiniLM-L6-v2` model to turn it into a vector.
4.  **Stores** all these vectors (and their corresponding text chunks) in a local `ChromaDB` database.

You only need to run this script **once** for each new PDF.

### 2. Querying (`query.py`)

This script is responsible for "asking" questions.
1.  **Loads** the existing `ChromaDB` database.
2.  **Loads** the same local embedding model.
3.  **Takes** your question from the command line (e.g., `"Who is Griffin?"`).
4.  **Retrieves** the most relevant text chunks from the database that match your question.
5.  **Generates** a prompt for the LLM that includes both your question and the retrieved context.
6.  **Passes** the final prompt to an LLM (configured for OpenAI) to get a natural language answer.

---

## 🛠️ Setup & Installation

Follow these steps to set up and run the project.

### 1. Prerequisites

* Python 3.10+
* Git (optional)

### 2. Install Dependencies

First, get the code and install the required Python libraries.

```bash
# Clone the repository (or download the files)
# git clone [https://github.com/your-username/pdf-rag-project.git](https://github.com/your-username/pdf-rag-project.git)
# cd pdf-rag-project

# It's highly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt
```

### 3. Add Your PDF

1.  Place your PDF file (e.g., `my-book.pdf`) inside the `/data` folder.
2.  Open `ingest.py` and update the `PDF_NAME` variable to match your file's name:

    ```python
    PDF_NAME = "my-book.pdf"  # <-- Change this to your file's name
    ```

### 4. Set Up API Key

This project uses a local model for embeddings (free) but is configured to use **OpenAI** for the final answer generation (which requires a paid API key).

1.  Create a file named `.env` in the root of the project folder.
2.  Add your OpenAI API key to this file:

    ```
    OPENAI_API_KEY="sk-..."
    ```

> **Note:** If you don't have a funded OpenAI account, the `query.py` script will fail with an `insufficient_quota` error. The `ingest.py` script will work perfectly, as it runs 100% locally.

---

## 🏃 Usage

### Step 1: Ingest Your Document

Run this command *once* to load your PDF into the vector database. This will create a `chroma_db` folder.

```bash
python ingest.py
```

*(The first time you run this, it will take a few moments to download the `all-MiniLM-L6-v2` embedding model from Hugging Face.)*

### Step 2: Ask a Question

Once ingestion is complete, you can ask questions about your document.

```bash
python query.py "What is the main theme of the document?"
```

**Example Questions (for "The Invisible Man"):**

```bash
python query.py "Who is the Invisible Man?"
python query.py "What is the name of the stranger?"
python query.py "What happened in the village of Iping?"
```
