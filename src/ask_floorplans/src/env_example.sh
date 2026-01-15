# Floor Plan AI System - Environment Configuration
# Copy this file to .env and fill in your values

# ============================================================================
# OpenAI Configuration (Required for AI features)
# ============================================================================
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-api-key-here

# ============================================================================
# Anthropic Configuration (Optional - for Claude integration)
# ============================================================================
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# ============================================================================
# Vector Database Configuration
# ============================================================================

# Pinecone (Production - Optional)
# Get your API key from: https://www.pinecone.io/
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=floor-plans

# FAISS (Local Development - No configuration needed)
FAISS_INDEX_PATH=./indexes/floor_plans.index

# Chroma (Alternative - No API key needed)
CHROMA_PERSIST_DIR=./chroma_db

# ============================================================================
# Graph Database Configuration (Optional)
# ============================================================================

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password-here

# ============================================================================
# Application Configuration
# ============================================================================

# Storage backend: faiss, pinecone, or chroma
VECTOR_STORE_TYPE=faiss

# Index name for vector storage
INDEX_NAME=floor-plans

# Enable graph database for relationship queries
USE_GRAPH_DB=false

# ============================================================================
# API Configuration (If running as web service)
# ============================================================================

# API host and port
API_HOST=0.0.0.0
API_PORT=8000

# Enable CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# API key for authentication (optional)
API_KEY=your-api-key-here

# ============================================================================
# Logging Configuration
# ============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log file path (optional)
LOG_FILE=./logs/floorplan_system.log

# ============================================================================
# Model Configuration
# ============================================================================

# Embedding model for vector search
# Options: all-MiniLM-L6-v2, all-mpnet-base-v2, sentence-transformers/all-roberta-large-v1
EMBEDDING_MODEL=all-MiniLM-L6-v2

# OpenAI model for generation
# Options: gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo
OPENAI_MODEL=gpt-4-turbo-preview

# Temperature for generation (0.0 to 1.0)
GENERATION_TEMPERATURE=0.7

# ============================================================================
# Performance Configuration
# ============================================================================

# Maximum number of results to return in searches
MAX_SEARCH_RESULTS=100

# Number of results for RAG context
RAG_CONTEXT_SIZE=5

# Cache size for floor plans
PLAN_CACHE_SIZE=1000

# Enable GPU acceleration (if available)
USE_GPU=false

# ============================================================================
# Development Configuration
# ============================================================================

# Enable debug mode
DEBUG=false

# Enable auto-reload in development
AUTO_RELOAD=true

# Enable request logging
LOG_REQUESTS=true

# ============================================================================
# Data Paths
# ============================================================================

# Directory for storing floor plans
PLANS_DIR=./data/plans

# Directory for storing indexes
INDEXES_DIR=./indexes

# Directory for exports
EXPORTS_DIR=./exports

# Directory for temporary files
TEMP_DIR=./temp
