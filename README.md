# Hybrid search rag

A RAG (Retrieval-Augmented Generation) pipeline that ingests PDFs including images and tables and answers questions using hybrid search combining BM25 and pgvector semantic search.

## Features

- **PDF ingestion** with `hi_res` OCR strategy via Unstructured
- **Image summarization** using GPT-4o vision images are described and stored alongside text
- **Hybrid retrieval** using Ensemble Retriever (BM25 + pgvector) with RRF merging
- **Multimodal prompting** — actual images passed to GPT-4o for visually-grounded answers
- **pgvector** as the vector store (via `langchain-postgres`)

## Architecture

```
PDF
 ↓
UnstructuredPDFLoader (hi_res, OCR)
 ↓
Elements: Text | Tables | Images
 ↓
Images → GPT-4o vision → summary + image_path stored
Text   → RecursiveCharacterTextSplitter (chunk_size=1500)
Tables → stored as-is
 ↓
OpenAI Embeddings → PGVector (PostgreSQL)

Query
 ↓
EnsembleRetriever
  ├── BM25 (keyword match)
  └── PGVector (semantic search)
 ↓
RRF merging → text chunks + image chunks
 ↓
Multimodal prompt (text context + base64 images) → GPT-4o → Answer
```

## Project Structure

```
hybrid_search_rag/
  ingestion.py       # PDF loading, chunking, image summarization, pgvector ingestion
  search_query.py    # hybrid retrieval, multimodal prompting, RAGAS evaluation
  main.ipynb         # notebook for running and testing
  requirements.txt   # dependencies
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Rajath1303/hybrid_search_rag
cd hybrid_search_rag
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

System dependencies for OCR:

```bash
# Mac
brew install tesseract poppler

# Ubuntu
sudo apt-get install tesseract-ocr poppler-utils
```

### 4. Start pgvector with Docker

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: vector
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```bash
docker-compose up -d
```

### 5. Set environment variables

```bash
# .env
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Ingest a PDF

```bash
python ingestion.py
# Enter PDF filepath: docs/book.pdf
```

### Query

```bash
python search_query.py
# Enter query: Who is O. Henry?
```

## Stack

| Component           | Technology                        |
| ------------------- | --------------------------------- |
| PDF Loader          | `unstructured` (hi_res, OCR)      |
| Image Understanding | GPT-4o vision                     |
| Embeddings          | `text-embedding-3-small` (OpenAI) |
| Vector Store        | pgvector (PostgreSQL)             |
| Sparse Retrieval    | BM25 (`rank-bm25`)                |
| Hybrid Retrieval    | LangChain EnsembleRetriever (RRF) |
| LLM                 | GPT-4o                            |
| Evaluation          | RAGAS                             |
| Framework           | LangChain                         |

## Evaluation Metrics (RAGAS)

| Metric            | Description                                    |
| ----------------- | ---------------------------------------------- |
| Faithfulness      | Is the answer grounded in retrieved context?   |
| Answer Relevancy  | Does the answer address the question?          |
| Context Precision | Are retrieved chunks relevant to the question? |
| Context Recall    | Was all necessary information retrieved?       |

## Requirements

```
langchain
langchain-openai
langchain-community
langchain-postgres
langchain-text-splitters
pgvector
psycopg
psycopg-binary
sqlalchemy
unstructured[pdf]
pillow
rank-bm25
datasets
python-dotenv
openai
```
