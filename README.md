# RAG using LangChain

A Retrieval-Augmented Generation (RAG) system built from scratch using LangChain, FAISS, Sentence Transformers, and Groq LLMs.

This project demonstrates the complete RAG pipeline, including document loading, chunking, embedding generation, vector storage, semantic retrieval, and LLM-powered response generation.

## Features

* Multi-format document loading

  * PDF
  * TXT
  * CSV
  * Excel (.xlsx)
  * Word (.docx)
  * JSON

* Document chunking using RecursiveCharacterTextSplitter

* Semantic embeddings using all-MiniLM-L6-v2

* FAISS vector database for efficient similarity search

* Metadata preservation for retrieved chunks

* Persistent vector store using FAISS and Pickle

* Groq-powered LLM integration using Llama 3.3 70B

* End-to-end Retrieval-Augmented Generation workflow

---

## Project Structure

```text
RAG/
├── data/
│   ├── pdf/
│   └── text_files/
│
├── notebook/
│   ├── document.ipynb
│   └── pdf_loader.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── embedding.py
│   ├── vectorstore.py
│   └── search.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Architecture

```text
Documents
    ↓
Document Loader
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Search
    ↓
Retrieved Context
    ↓
Groq LLM
    ↓
Generated Response
```

---

## Workflow

### 1. Document Loading

The system loads documents from the data directory and converts them into LangChain Document objects.

Supported formats:

* PDF
* TXT
* CSV
* XLSX
* DOCX
* JSON

### 2. Chunking

Documents are split into smaller overlapping chunks using:

```python
chunk_size = 1000
chunk_overlap = 200
```

This preserves context across chunk boundaries.

### 3. Embedding Generation

Embeddings are generated using:

```text
all-MiniLM-L6-v2
```

Characteristics:

* Lightweight
* Fast inference
* 384-dimensional embeddings

### 4. Vector Storage

Embeddings are stored in a FAISS IndexFlatL2 index.

Metadata is stored separately and linked to vectors using index positions.

### 5. Retrieval

User queries are embedded using the same embedding model.

FAISS performs nearest-neighbor search and returns the most relevant chunks.

### 6. Generation

Retrieved chunks are combined into a context prompt and passed to a Groq-hosted LLM for response generation.

---

## Technologies Used

### Frameworks & Libraries

* LangChain
* FAISS
* Sentence Transformers
* NumPy
* Pickle
* python-dotenv

### Embedding Model

* all-MiniLM-L6-v2

### LLM

* Llama 3.3 70B
* Groq API

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sreedevi101010/RAG-using-LangChain.git
cd RAG-using-LangChain
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Project

Build the vector store:

```bash
python -m src.vectorstore
```

Run retrieval and summarization:

```bash
python -m src.search
```

---

## Example Query

```text
What is machine learning?
```

### Retrieval

The system retrieves the most relevant document chunks using semantic similarity search.

### Generation

The retrieved context is passed to the LLM to generate a concise response grounded in the source documents.

---

## Learning Objectives

This project was built to gain a deeper understanding of:

* Retrieval-Augmented Generation (RAG)
* Document chunking strategies
* Embedding generation
* Vector databases
* Semantic search
* LangChain internals
* FAISS indexing and retrieval
* LLM integration
* End-to-end AI application development

---

## Future Improvements

* Hybrid Search (BM25 + Dense Retrieval)
* Cross-Encoder Re-ranking
* Metadata-based Filtering
* Multi-Query Retrieval
* Streamlit Web Interface
* Conversational Memory
* Source Citation Support
* Advanced Evaluation Metrics

---

## Author

**Sreedevi K**

Computer Science Engineering Student | AI & NLP Enthusiast
