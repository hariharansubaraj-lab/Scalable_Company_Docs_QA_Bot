# 🏢 Scalable Company Documents Q&A Bot (Enterprise RAG)

A scalable, production-grade Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Milvus Vector Database v3.0.1**, **Sentence Transformers**, **Gemini LLM**, and **Streamlit**. Designed for high-concurrency enterprise PDF ingestion, hybrid search, cross-encoder reranking, and contextual question answering.

---

## 🌟 Key Features

- **Scalable Vector Infrastructure**: Milvus v3.0.1 standalone architecture backed by MinIO distributed object storage and Etcd key-value coordinator.
- **Enterprise PDF Processing**: Ingestion pipeline for parsing company policies and employee handbooks with recursive chunking and source metadata retention.
- **Hybrid Retrieval Pipeline**: Combines dense semantic vector retrieval (via `all-MiniLM-L6-v2` in Milvus) with sparse keyword matching (BM25) for precision and recall.
- **Cross-Encoder Reranking**: Re-ranks candidate passages using `BAAI/bge-reranker-base` to surface the most relevant context before LLM synthesis.
- **Context-Aware Query Rewriting**: Rewrites conversational multi-turn queries using Gemini to maintain context across follow-up questions.
- **Document Scoping & Filtering**: Supports querying across all ingested documents or narrowing search scope to specific document IDs.
- **Interactive Streamlit UI**: User-friendly web interface with document selector, PDF upload, real-time response generation, and source citations.
- **FastAPI Microservice**: High-throughput REST API with validation, health checks, and modular routing.
- **Automated RAG Evaluation**: Built-in benchmark suite to evaluate retrieval accuracy and generation quality.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Document Ingestion Pipeline"]
        PDF[PDF Documents] --> Loader[Document Loader]
        Loader --> Chunker[Text Chunker (Size: 500, Overlap: 50)]
        Chunker --> Embedder[Sentence Transformer (all-MiniLM-L6-v2)]
        Embedder --> Milvus[(Milvus Vector DB v3.0)]
    end

    subgraph QueryFlow["Query & Retrieval Flow"]
        UserQuery[User Query & History] --> Rewriter[Gemini Query Rewriter]
        Rewriter --> DenseSearch[Dense Vector Search]
        Rewriter --> BM25Search[BM25 Keyword Search]
        Milvus --> DenseSearch
        DenseSearch & BM25Search --> HybridRetriever[Hybrid Fusion]
        HybridRetriever --> CrossEncoder[BAAI/bge-reranker-base Reranker]
        CrossEncoder --> TopChunks[Top-K Re-ranked Context]
    end

    subgraph Generation["Response Generation"]
        TopChunks & UserQuery --> Prompt[Contextual Prompt Builder]
        Prompt --> Gemini[Gemini LLM]
        Gemini --> Response[Answer with Page Citations]
    end
```

---

## 📁 Repository Structure

```
Scalable_Company_Docs_QA_Bot/
├── app/
│   ├── __init__.py           # Package marker
│   ├── bm25_retriever.py     # BM25 sparse keyword retriever
│   ├── chunking.py           # Recursive document chunking
│   ├── config.py             # Configuration & environment loader
│   ├── document_loader.py    # PDF text extraction & cleaning
│   ├── embeddings.py         # Embedding model initialization
│   ├── llm.py                # Gemini LLM integration & prompt template
│   ├── main.py               # FastAPI application & REST endpoints
│   ├── query_rewriter.py     # Contextual query reformulation
│   ├── reranker.py           # Cross-encoder reranking (BGE reranker)
│   ├── retriever.py          # Hybrid retrieval coordinator
│   └── vector_store.py       # Milvus collection schema & vector search
├── data/
│   ├── company_policies.pdf  # Sample enterprise policy document
│   └── employee_handbook.pdf # Sample employee handbook document
├── evaluation/
│   ├── evaluation.py         # RAG evaluation runner
│   └── evaluation_questions.json # Evaluation question-answer benchmark
├── .env.example              # Environment variables template
├── .gitignore                # Excludes secrets, database volumes, & cache
├── docker-compose.yml        # Milvus v3.0, Etcd v3.5, & MinIO service stack
├── embedEtcd.yaml            # Etcd configuration
├── requirements.txt          # Python dependencies
├── reset_milvus.py           # Utility to clear/reset Milvus collection
├── standalone.bat            # Windows service manager script
├── streamlit_app.py          # Streamlit frontend user interface
├── test_loader.py            # Diagnostic script to test document loading
└── user.yaml                 # User configuration
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.12)
- **Docker & Docker Compose**: For running Milvus, MinIO, and Etcd

### 2. Clone and Setup Environment

```bash
git clone https://github.com/hariharansubaraj-lab/Scalable_Company_Docs_QA_Bot.git
cd Scalable_Company_Docs_QA_Bot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `.env` from the provided `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and set your Google Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 4. Start Milvus Infrastructure

Start Milvus standalone, Etcd, and MinIO via Docker Compose:

```bash
docker-compose up -d
```

Verify services are healthy:
```bash
docker-compose ps
```

### 5. Start the FastAPI Backend

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at: `http://127.0.0.1:8000/docs`

### 6. Launch the Streamlit Frontend

In a separate terminal:

```bash
streamlit run streamlit_app.py
```
Access the application at `http://localhost:8501`.

---

## 📊 Evaluation & Testing

Run the automated evaluation benchmark:

```bash
python -m evaluation.evaluation
```

To test document loading independently:

```bash
python test_loader.py
```

To reset the Milvus collection:

```bash
python reset_milvus.py
```

---

## 🛡️ Security & Best Practices

- Real API credentials in `.env` are strictly excluded from git tracking via `.gitignore`.
- Milvus database data and storage volumes (`volumes/`, `*.sst`, `*.log`) are ignored to prevent large binary blobs in the repository history.
- An sanitized `.env.example` is provided for reproducibility.
