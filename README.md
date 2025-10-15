# 🚀 Vectron - Rust-Based Vector Database with Embedded AI

![Rust](https://img.shields.io/badge/Rust-🦀-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

A high-performance, lightweight vector database built in Rust with text-to-embedding conversion, optimized vector storage, and similarity search capabilities. Vectron combines the speed of Rust with the power of modern AI embedding models.




## ✨ Features

- **🧠 Multi-Model Embedding Support**:
  - ✅ Local models (Sentence Transformers: MiniLM, BGE, etc.)
  - ✅ Remote APIs (OpenAI, Cohere, HuggingFace)
  - ✅ Model registry with configuration
  - ✅ Benchmarking and model comparison
- **🔀 Decoupled Architecture**:
  - ✅ Microservice for embedding generation
  - ✅ Standalone vector database service
  - ✅ Dockerized for easy deployment
- **⚡ High-Performance Vector Storage**: Fast in-memory storage with persistence capabilities
- **🔍 Similarity Search**: Efficient cosine similarity search with top-K results
- **🔄 Comprehensive REST API**: Full CRUD operations for vectors and embeddings
- **💾 Persistence Layer**: Save and load vectors to/from disk automatically
- **📊 Built-in Benchmarking**: Measure performance of vector operations and embedding models

## 🏗️ Architecture

```
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    🚀 VECTRON VECTOR DATABASE - OVERVIEW ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                CLIENT LAYER                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                     │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│    │    cURL     │    │   Postman   │    │ React Web   │    │   Python    │    │   Custom    │                   │
│    │             │    │             │    │  Dashboard  │    │   Client    │    │    Apps     │                   │
│    │ CLI Testing │    │ API Testing │    │ Port 3001   │    │             │    │             │                   │
│    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘                   │
│           │                   │                   │                   │                   │                       │
│           └───────────────────┼───────────────────┼───────────────────┼───────────────────┘                       │
│                               │                   │                   │                                           │
│                               │         HTTP/REST API Calls           │                                           │
│                               │                   │                   │                                           │
└───────────────────────────────┼───────────────────┼───────────────────┼───────────────────────────────────────────┘
                                │                   │                   │
                                ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              CORE SERVICES                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    🦀 VECTRON RUST SERVICE                                                    │ │
│  │                                         Port 3000                                                             │ │
│  │                                                                                                                │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                                    REST API LAYER                                                         │ │ │
│  │  │                                                                                                            │ │ │
│  │  │  GET  /                    POST /vector              POST /search/vector                                  │ │ │
│  │  │  GET  /vector/:id          DELETE /vector/:id        POST /search/text                                   │ │ │
│  │  │  GET  /vectors             POST /embed               POST /benchmark                                      │ │ │
│  │  │  POST /upsert-text         ...                       ...                                                  │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                    │                                                           │ │
│  │                                                    ▼                                                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                                 VECTOR DATABASE ENGINE                                                    │ │ │
│  │  │                                                                                                            │ │ │
│  │  │  • In-Memory Storage: HashMap<String, Vec<f32>>                                                           │ │ │
│  │  │  • Cosine Similarity Search: O(n*d) linear scan                                                          │ │ │
│  │  │  • CRUD Operations: Insert, Get, Delete, List                                                            │ │ │
│  │  │  • Thread-Safe: Arc<RwLock<VectorStore>>                                                                  │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                    │                                                           │ │
│  │                                                    ▼                                                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                                  PERSISTENCE LAYER                                                        │ │ │
│  │  │                                                                                                            │ │ │
│  │  │  • JSON File Storage: ./data/index.json                                                                   │ │ │
│  │  │  • Auto-Save on Changes                                                                                   │ │ │
│  │  │  • Load on Startup                                                                                        │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                    │                                                               │
│                                                    │ HTTP Requests                                                 │
│                                                    ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                🐍 PYTHON EMBEDDING SERVICE                                                    │ │
│  │                                         Port 8000                                                             │ │
│  │                                                                                                                │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                                   FASTAPI ENDPOINTS                                                       │ │ │
│  │  │                                                                                                            │ │ │
│  │  │  GET  /health              POST /embed                POST /benchmark                                     │ │ │
│  │  │  GET  /models              POST /batch-embed          POST /auth/token                                   │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                    │                                                           │ │
│  │                                                    ▼                                                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                                  EMBEDDING MODELS                                                         │ │ │
│  │  │                                                                                                            │ │ │
│  │  │  🤗 Local Models (Sentence Transformers):                                                                 │ │ │
│  │  │     • MiniLM (384d) - Fast, lightweight                                                                   │ │ │
│  │  │     • BGE-Small (384d) - Balanced performance                                                             │ │ │
│  │  │     • BGE-Base (768d) - Higher quality                                                                    │ │ │
│  │  │                                                                                                            │ │ │
│  │  │  🌐 Remote APIs:                                                                                           │ │ │
│  │  │     • OpenAI (1536d) - High quality, paid                                                                 │ │ │
│  │  │     • Cohere (384d) - Enterprise grade                                                                    │ │ │
│  │  │     • HuggingFace (768d) - Open source models                                                             │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              STORAGE LAYER                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                        💾 FILE SYSTEM                                                         │ │
│  │                                                                                                                │ │
│  │  ./data/                                                                                                       │ │
│  │  ├── index.json              ← Vector storage (JSON format)                                                   │ │
│  │  └── benchmark_temp/         ← Temporary benchmark data                                                       │ │
│  │                                                                                                                │ │
│  │  ./model_cache/                                                                                               │ │
│  │  ├── sentence-transformers/  ← Cached local models                                                            │ │
│  │  └── huggingface/           ← Downloaded model files                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            🔄 DATA FLOW OVERVIEW                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                      📝 TEXT INSERTION FLOW                                                   │ │
│  │                                                                                                                │ │
│  │  Client Text ──→ POST /upsert-text ──→ Rust Service ──→ HTTP /embed ──→ Python Service                      │ │
│  │       │                                     ▲                                    │                            │ │
│  │       │                                     │                                    ▼                            │ │
│  │       │                                Vector Storage ←──── Vector ←──── Embedding Model                     │ │
│  │       │                                     │                                                                 │ │
│  │       └─────────────────── Success Response ←─────────────────────────────────────┘                         │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                      🔍 SEARCH FLOW                                                           │ │
│  │                                                                                                                │ │
│  │  Query Text ──→ POST /search/text ──→ Generate Embedding ──→ Cosine Similarity ──→ Ranked Results            │ │
│  │       │                                       │                        │                      │               │ │
│  │       │                                       ▼                        ▼                      │               │ │
│  │       │                              Python Service ──→ All Stored Vectors ──→ Top-K Matches │               │ │
│  │       │                                                                                        │               │ │
│  │       └─────────────────────────────── Search Results ←───────────────────────────────────────┘               │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    ⚡ DIRECT VECTOR OPERATIONS                                                │ │
│  │                                                                                                                │ │
│  │  Vector Data ──→ POST /vector ──→ HashMap Storage ──→ Optional Persistence ──→ Success Response              │ │
│  │                                                                                                                │ │
│  │  Vector ID ──→ GET /vector/:id ──→ HashMap Lookup ──→ Vector Data Response                                    │ │
│  │                                                                                                                │ │
│  │  Vector ID ──→ DELETE /vector/:id ──→ HashMap Remove ──→ Optional Persistence ──→ Success Response           │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          🐳 DEPLOYMENT ARCHITECTURE                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                      DOCKER COMPOSE SETUP                                                     │ │
│  │                                                                                                                │ │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                                           │ │
│  │  │   dashboard     │    │    vectron      │    │ embedding-service│                                           │ │
│  │  │                 │    │                 │    │                 │                                           │ │
│  │  │ React + Vite    │    │ Rust + Axum     │    │ Python + FastAPI│                                           │ │
│  │  │ Port 3001       │    │ Port 3000       │    │ Port 8000       │                                           │ │
│  │  │                 │    │                 │    │                 │                                           │ │
│  │  │ Web Interface   │    │ Vector Database │    │ ML Models       │                                           │ │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘                                           │ │
│  │           │                       │                       │                                                   │ │
│  │           │                       │                       │                                                   │ │
│  │           └───────────────────────┼───────────────────────┘                                                   │ │
│  │                                   │                                                                           │ │
│  │                                   ▼                                                                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                                    SHARED VOLUMES                                                         │ │ │
│  │  │                                                                                                            │ │ │
│  │  │  ./data:/vectron/data              ← Vector persistence                                                   │ │ │
│  │  │  ./model_cache:/app/model_cache    ← ML model caching                                                     │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            🎯 KEY FEATURES                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                     │
│  ✅ Multi-Model Embedding Support    ✅ High-Performance Vector Search    ✅ RESTful API Interface                │
│  ✅ Persistent Storage               ✅ Thread-Safe Operations            ✅ Docker Containerization              │
│  ✅ Benchmarking & Monitoring        ✅ Fallback Mechanisms              ✅ Web Dashboard                         │
│  ✅ Authentication & API Keys        ✅ Batch Processing                  ✅ Model Comparison                      │
│                                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          📊 PERFORMANCE CHARACTERISTICS                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                     │
│  • Vector Storage: O(1) insert/get/delete operations                                                              │
│  • Similarity Search: O(n*d) linear scan (n=vectors, d=dimensions)                                                │
│  • Memory Usage: ~4 bytes per dimension per vector + HashMap overhead                                             │
│  • Persistence: JSON serialization (human-readable but not optimized for large datasets)                         │
│  • Concurrency: Read-write locks allow multiple concurrent readers                                                 │
│  • Embedding Generation: Depends on model (local: ~10-50ms, remote: ~100-500ms)                                  │
│                                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```


## 📊 Model Benchmark Results

Example benchmark comparing embedding models:

| Model ID    | Provider           | Dimensions | Processing Time (s) | Contrast Score |
|-------------|-------------------|------------|---------------------|----------------|
| minilm      | sentence-transformers | 384     | 0.012              | 0.068          |
| bge-small   | sentence-transformers | 384     | 0.015              | 0.082          |
| openai-small| openai            | 1536       | 0.321              | 0.103          |
| cohere-english | cohere         | 384        | 0.456              | 0.089          |

## 🚀 Getting Started

### Running with Docker

The easiest way to get started is using Docker Compose:

```bash
docker-compose up
```

This will start both the Vectron database and the embedding service.

### Manual Setup

1. Clone the repository
2. Install Python dependencies:

```bash
cd python/embedding_service
pip install -r requirements.txt
```

3. Copy the environment template and add your API keys:

```bash
cp env.template .env
# Edit .env with your OpenAI, Cohere, and HuggingFace API keys
```

4. Start the embedding service:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

5. In a new terminal, build and run Vectron:

```bash
cargo run --release
```

The Vectron API will be available at `http://localhost:3000` and the embedding service at `http://localhost:8000`.

## 📚 API Usage Examples

### Insert a vector from text

```bash
curl -X POST http://localhost:3000/upsert-text \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "text": "This is a sample document for vector search"}'
```

### Search similar vectors by text

```bash
curl -X POST http://localhost:3000/search/text \
  -H "Content-Type: application/json" \
  -d '{"text": "sample document", "model_id": "openai-small"}' \
  -G -d 'top_k=3'
```

### List available embedding models

```bash
curl http://localhost:8000/models
```

### Benchmark embedding models

```bash
./benchmark_models.sh
```

## 🧪 Development Roadmap

This project has been developed in phases:

- ✅ Phase 0: Project Setup - Initial project structure
- ✅ Phase 1: Embedding Service - Decoupled embedding microservice
- ✅ Phase 2: Multi-Model Support - Multiple embedding providers and benchmarking
- 🔄 Phase 3: Advanced Vector Storage - Optimized in-memory storage with persistence
- 🔄 Phase 4: Search Algorithms - Similarity search optimization
- 🔄 Phase 5: Advanced Features - Clustering and segmentation
- 🔄 Phase 6: Performance Tuning - Testing and benchmarking

## 🔧 Future Enhancements

- Approximate search using HNSW or similar algorithms
- Clustering and visualization tools
- Authentication and access control
- Distributed vector storage for scaling
- Web UI for interactive demos



## 🙏 Acknowledgements

- HuggingFace for their sentence-transformers models
- OpenAI, Cohere, and HuggingFace for embedding APIs
- The Rust community for excellent libraries and tools
