# 🤖 Mini Knowledge Assistant with LangGraph + RAG

A sophisticated backend service that combines **LangGraph** and **RAG** (Retrieval Augmented Generation) to create an intelligent knowledge assistant capable of answering questions based on uploaded documents.

## 🏗️ Architecture

![High-Level Architecture](High-Level%20Architecture.png)

### 🏛️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🌐 FastAPI REST API                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  📡 Endpoints: /upload, /chat, /health                                     │
│  🔧 Middleware: CORS, Request Validation, Error Handling                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        🧠 LangGraph Workflow Engine                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  📋 State Management: Question, Context, Answer, Execution Log             │
│  🔀 Conditional Branching: Context Quality Detection                       │
│  ⚡ Parallel Execution: Vector + Keyword Search                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          🔍 RAG Service Layer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  📄 Document Processing: PDF, Markdown, Text                               │
│  🔤 Text Chunking: RecursiveCharacterTextSplitter                          │
│  🧮 Embeddings: Nomic AI (nomic-embed-text-v1.5)                          │
│  🔍 Retrieval: Vector Similarity + Keyword Search                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        💾 Vector Database (Chroma)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  📊 Document Storage: Chunked documents with metadata                      │
│  🔍 Similarity Search: Vector-based retrieval                              │
│  💿 Persistence: Local file system storage                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        🤖 LLM Service (Groq)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  🧠 Model: Llama 3.3 70B Versatile                                        │
│  ⚡ Speed: High-performance inference                                      │
│  🎯 Temperature: 0 (deterministic responses)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔄 LangGraph Workflow Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  📝 Question    │───▶│  🔍 Validate     │───▶│  🔍 Retrieve    │
│  Input          │    │  Question        │    │  Context        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  ⚡ Parallel    │
                                               │  Execution     │
                                               └─────────────────┘
                                                        │
                                    ┌───────────────────┼───────────────────┐
                                    ▼                   ▼                   ▼
                            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                            │  🔍 Vector  │    │  🔤 Keyword │    │  🔗 Merge   │
                            │  Search     │    │  Search     │    │  Results    │
                            └─────────────┘    └─────────────┘    └─────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  🤔 Context    │
                                               │  Quality Check │
                                               └─────────────────┘
                                                        │
                                    ┌───────────────────┼───────────────────┐
                                    ▼                   ▼                   ▼
                            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                            │  ✅ Sufficient │    │  ❌ Insufficient │    │  🎯 Generate │
                            │  Context     │    │  Context     │    │  Answer     │
                            └─────────────┘    └─────────────┘    └─────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  📝 Format     │
                                               │  Response      │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  📤 Final      │
                                               │  Response      │
                                               └─────────────────┘
```

### 🔧 Component Details

#### **🌐 FastAPI Layer**
- **REST API**: Clean, well-documented endpoints
- **Request Validation**: Pydantic models for type safety
- **Error Handling**: Comprehensive HTTPException handling
- **CORS Support**: Ready for frontend integration

#### **🧠 LangGraph Engine**
- **State Management**: Comprehensive state with execution tracking
- **Conditional Branching**: Smart context quality detection
- **Parallel Execution**: True async parallel processing
- **Workflow Orchestration**: Complete pipeline management

#### **🔍 RAG Service**
- **Document Processing**: Multi-format support (PDF, Markdown, Text)
- **Text Chunking**: Intelligent document segmentation
- **Embeddings**: High-quality vector representations
- **Dual Retrieval**: Vector similarity + keyword search

#### **💾 Vector Database**
- **Chroma**: Local, persistent vector storage
- **Similarity Search**: Fast vector-based retrieval
- **Metadata**: Rich document and chunk information
- **Persistence**: Reliable data storage

#### **🤖 LLM Service**
- **Groq**: High-speed inference platform
- **Llama 3.3**: Latest 70B parameter model
- **Performance**: Optimized for speed and accuracy
- **Reliability**: Enterprise-grade service

## ✨ Features

### 🔧 Core Functionality
- **Document Upload & Processing**: Supports text files, Markdown, and comprehensive PDF processing
- **Intelligent Question Answering**: Uses RAG to provide context-aware responses
- **Multi-Modal Retrieval**: Combines vector similarity search with keyword matching
- **Robust PDF Support**: Dual extraction methods (pdfplumber + PyPDF2) for maximum compatibility

### 🧠 LangGraph Workflow
- **Conditional Branching**: Automatically detects when context is insufficient and provides fallback responses
- **Parallel Execution**: Runs vector and keyword searches simultaneously for optimal performance
- **State Management**: Maintains conversation context and retrieved results
- **Error Handling**: Graceful degradation when services are unavailable

### 🌐 API Features
- **RESTful Design**: Clean, well-documented endpoints
- **Async Processing**: Non-blocking document processing and query handling
- **Health Checks**: Built-in monitoring endpoints
- **CORS Support**: Ready for frontend integration

1. 
   cd Dynafy_Task
   

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: This will install PDF processing libraries (pypdf2, pdfplumber) for comprehensive document support.

4. **Configure environment variables**
   ```bash
   # Copy the example file
   cp env.example .env
   
   # Edit .env and add your API keys:
   # GROQ_API_KEY=your_groq_api_key_here
   # NOMIC_API_KEY=your_nomic_api_key_here
   ```

5. **Run the server**
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

### 🔑 API Keys Required

#### **Groq API Key**
- **Purpose**: LLM inference using Llama 3.3 70B model
- **Get API Key**: [Groq Console](https://console.groq.com/)
- **Free Tier**: 14,400 requests/day, 30 requests/minute
- **Usage**: High-speed inference with low latency
- **Environment Variable**: `GROQ_API_KEY`

#### **Nomic API Key**
- **Purpose**: High-quality embeddings for document similarity search
- **Get API Key**: [Nomic AI](https://atlas.nomic.ai/)
- **Free Tier**: 100,000 tokens/month
- **Usage**: Document embedding generation and vector search
- **Environment Variable**: `NOMIC_API_KEY`

#### **Optional Configuration**
- **Chroma Persist Directory**: `CHROMA_PERSIST_DIR` (default: `./data/chroma`)
- **Custom Vector Store**: Configure persistent storage location

#### **Environment Setup**
```bash
# Create .env file in project root
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
echo "NOMIC_API_KEY=your_nomic_api_key_here" >> .env
echo "CHROMA_PERSIST_DIR=./data/chroma" >> .env
```

**Note**: Both services offer generous free tiers perfect for development and testing.

## 📡 API Usage

The API provides three main endpoints:

### 🏠 Health Check
```bash
curl -X GET "http://localhost:8000/"
```
**Response:**
```json
{"message": "Mini Knowledge Assistant is running!"}
```

### 📄 Upload Documents (Single or Multiple)
```bash
# Upload a single document
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@examples/sample.txt"

# Upload multiple documents
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@examples/sample.txt" \
  -F "files=@examples/comprehensive_guide.md" \
  -F "files=@examples/USAMA-YASEEN.pdf"
```
**Response (Single Document):**
```json
{
  "message": "Successfully processed all 1 documents.",
  "successful_uploads": [
    {
      "filename": "sample.txt",
      "status": "success"
    }
  ],
  "total_processed": 1,
  "success_count": 1,
  "failure_count": 0
}
```
**Response (Multiple Documents):**
```json
{
  "message": "Successfully processed all 3 documents.",
  "successful_uploads": [
    {
      "filename": "sample.txt",
      "status": "success"
    },
    {
      "filename": "comprehensive_guide.md",
      "status": "success"
    },
    {
      "filename": "USAMA-YASEEN.pdf",
      "status": "success"
    }
  ],
  "total_processed": 3,
  "success_count": 3,
  "failure_count": 0
}
```
**Response (Partial Success):**
```json
{
  "message": "Processed 2 documents successfully, 1 failed.",
  "successful_uploads": [
    {
      "filename": "sample.txt",
      "status": "success"
    },
    {
      "filename": "comprehensive_guide.md",
      "status": "success"
    }
  ],
  "failed_uploads": [
    {
      "filename": "invalid_file.exe",
      "status": "failed",
      "error": "Unsupported file type: invalid_file.exe"
    }
  ],
  "total_processed": 3,
  "success_count": 2,
  "failure_count": 1
}
```

**Supported file types:** `.txt`, `.md`, `.markdown`, `.pdf` (full support with dual extraction methods)

### 💬 Ask Questions
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main applications of AI mentioned in the document?"}'
```
**Response:**
```json
{
  "response": "Based on the uploaded documents, the main applications of AI include healthcare diagnostics, autonomous vehicles, natural language processing, and financial fraud detection..."
}
```

### 🔧 Interactive API Documentation
Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎯 Design Choices & Architecture Decisions

### 🗄️ **Vector Store - Chroma**
- **Why**: Excellent local development experience, no external dependencies
- **Benefits**: Fast similarity search, persistent storage, easy integration with LangChain
- **Trade-offs**: Single-node deployment (suitable for demo/development)

### 🧮 **Embeddings - Nomic AI**
- **Why**: High-quality embeddings with generous free tier
- **Benefits**: Optimized for retrieval tasks, good performance on diverse content
- **Alternative**: Could easily switch to OpenAI embeddings if needed

### 🤖 **LLM - Groq (Llama 3.3)**
- **Why**: Extremely fast inference, cost-effective, good quality responses
- **Benefits**: Low latency, handles context well, reliable API
- **Trade-offs**: Dependent on external service (could fallback to local models)

### 🔀 **LangGraph Workflow Design**
```python
# Key workflow features:
- Parallel Execution: Vector + Keyword search run simultaneously
- Conditional Logic: Smart fallback when context is insufficient  
- State Management: Maintains context and error states
- Error Resilience: Graceful degradation on component failures
```

### 📝 **Text Processing Strategy**
- **Chunking**: RecursiveCharacterTextSplitter with 1000 char chunks, 200 char overlap
- **Why**: Balances context preservation with retrieval granularity
- **File Support**: Extensible loader system for multiple file types

## 📁 Project Structure

```
Dynafy_Task/
├── 📂 src/                          # Source code
│   ├── 📄 main.py                   # FastAPI application entry point
│   ├── 📄 models.py                 # Pydantic data models
│   ├── 📂 services/                 # Business logic layer
│   │   ├── 📄 rag_service.py        # RAG implementation & vector operations
│   │   ├── 📄 graph_service.py      # LangGraph workflow orchestration
│   │   └── 📄 embeddings.py         # Nomic embeddings service
│   ├── 📂 routes/                   # API route handlers (future expansion)
│   │   ├── 📄 upload.py             # Document upload endpoints
│   │   └── 📄 chat.py               # Chat/query endpoints
│   └── 📂 utils/                    # Utility functions
│       └── 📄 file_loader.py        # Multi-format file processing
├── 📂 data/                         # Persistent data storage
│   └── 📂 chroma/                   # Vector database files
├── 📂 examples/                     # Test documents & sample queries
│   ├── 📄 sample.txt                # Sample document for testing
│   └── 📄 example_queries.txt       # Pre-written test questions
├── 📄 requirements.txt              # Python dependencies
├── 📄 env.example                   # Environment variables template
├── 📄 README.md                     # This documentation
└── 📄 test.py                       # Basic database inspection scrip
```

### 🏗️ Architecture Layers
- **API Layer** (`main.py`): FastAPI routes and middleware
- **Service Layer** (`services/`): Core business logic and external integrations  
- **Utility Layer** (`utils/`): Reusable helper functions
- **Data Layer** (`data/`): Persistent storage and vector database

## 🧪 Testing Guide

### Quick Start Test
1. **Start the server**
   ```bash
   uvicorn src.main:app --reload
   ```

2. **Upload sample document**
   ```bash
   curl -X POST "http://localhost:8000/upload" \
     -F "file=@examples/sample.txt"
   ```

3. **Test with example queries**
   ```bash
   # Test successful retrieval
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What are the main applications of AI?"}'
   
   # Test fallback behavior
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is quantum computing?"}'
   ```

### 📋 Test Scenarios

| Test Case | Expected Behavior |
|-----------|-------------------|
| Upload valid document | ✅ Success message |
| Upload unsupported file | ❌ Error with file type message |
| Query about uploaded content | ✅ Relevant answer from document |
| Query about unrelated topic | ⚠️ Fallback "no information found" message |
| Empty/malformed query | ❌ Validation error |

### 📁 Example Files
- `examples/sample.txt`: AI overview document for testing
- `examples/example_queries.txt`: Pre-written test questions

### 🔍 Debugging
- Check logs for LangGraph workflow execution
- Use FastAPI docs at `/docs` for interactive testing