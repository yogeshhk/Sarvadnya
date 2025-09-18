"""
Configuration management for the Financial RAG system
Handles environment variables, model settings, and system parameters
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_chunk_size: int = 1000
    text_overlap: int = 200
    embedding_dim: Optional[int] = None

@dataclass
class VectorStoreConfig:
    """Configuration for vector storage"""
    collection_name: str = "financial_docs"
    persist_directory: str = "./chroma_db"
    distance_metric: str = "cosine"
    sql_db_path: str = "./financial_tables.db"

@dataclass
class LLMConfig:
    """Configuration for language models"""
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 4000
    api_key: Optional[str] = None
    vision_model: str = "gpt-4-vision-preview"

@dataclass
class AgentConfig:
    """Configuration for the agent system"""
    max_retrieval_chunks: int = 8
    max_table_results: int = 10
    sql_confidence_threshold: float = 0.5
    context_expansion_threshold: int = 3

@dataclass
class SystemConfig:
    """Main system configuration"""
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    llm: LLMConfig
    agent: AgentConfig
    debug: bool = False
    log_level: str = "INFO"

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self._config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file and environment"""
        # Default configuration
        config_dict = {
            "embedding": {
                "text_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                "text_chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
                "text_overlap": int(os.getenv("CHUNK_OVERLAP", "200"))
            },
            "vector_store": {
                "collection_name": os.getenv("COLLECTION_NAME", "financial_docs"),
                "persist_directory": os.getenv("VECTOR_DB_PATH", "./chroma_db"),
                "distance_metric": os.getenv("DISTANCE_METRIC", "cosine"),
                "sql_db_path": os.getenv("SQL_DB_PATH", "./financial_tables.db")
            },
            "llm": {
                "model_name": os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("MAX_TOKENS", "4000")),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "vision_model": os.getenv("VISION_MODEL", "gpt-4-vision-preview")
            },
            "agent": {
                "max_retrieval_chunks": int(os.getenv("MAX_RETRIEVAL_CHUNKS", "8")),
                "max_table_results": int(os.getenv("MAX_TABLE_RESULTS", "10")),
                "sql_confidence_threshold": float(os.getenv("SQL_CONFIDENCE", "0.5")),
                "context_expansion_threshold": int(os.getenv("CONTEXT_THRESHOLD", "3"))
            },
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        }
        
        # Load from config file if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    self._merge_configs(config_dict, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_path}: {e}")
        
        # Create configuration objects
        return SystemConfig(
            embedding=EmbeddingConfig(**config_dict["embedding"]),
            vector_store=VectorStoreConfig(**config_dict["vector_store"]),
            llm=LLMConfig(**config_dict["llm"]),
            agent=AgentConfig(**config_dict["agent"]),
            debug=config_dict["debug"],
            log_level=config_dict["log_level"]
        )
    
    def _merge_configs(self, base_config: Dict, file_config: Dict):
        """Merge file configuration with base configuration"""
        for key, value in file_config.items():
            if key in base_config:
                if isinstance(value, dict) and isinstance(base_config[key], dict):
                    base_config[key].update(value)
                else:
                    base_config[key] = value
    
    @property
    def config(self) -> SystemConfig:
        """Get the current configuration"""
        return self._config
    
    def save_config(self):
        """Save current configuration to file"""
        config_dict = {
            "embedding": {
                "text_model": self._config.embedding.text_model,
                "text_chunk_size": self._config.embedding.text_chunk_size,
                "text_overlap": self._config.embedding.text_overlap
            },
            "vector_store": {
                "collection_name": self._config.vector_store.collection_name,
                "persist_directory": self._config.vector_store.persist_directory,
                "distance_metric": self._config.vector_store.distance_metric,
                "sql_db_path": self._config.vector_store.sql_db_path
            },
            "llm": {
                "model_name": self._config.llm.model_name,
                "temperature": self._config.llm.temperature,
                "max_tokens": self._config.llm.max_tokens,
                "vision_model": self._config.llm.vision_model
                # Note: API key is not saved for security
            },
            "agent": {
                "max_retrieval_chunks": self._config.agent.max_retrieval_chunks,
                "max_table_results": self._config.agent.max_table_results,
                "sql_confidence_threshold": self._config.agent.sql_confidence_threshold,
                "context_expansion_threshold": self._config.agent.context_expansion_threshold
            },
            "debug": self._config.debug,
            "log_level": self._config.log_level
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update_config(self, **updates):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def validate_config(self) -> bool:
        """Validate the current configuration"""
        errors = []
        
        # Check API key
        if not self._config.llm.api_key:
            errors.append("OpenAI API key is required")
        
        # Check model names
        if not self._config.embedding.text_model:
            errors.append("Text embedding model is required")
        
        # Check paths
        if not self._config.vector_store.persist_directory:
            errors.append("Vector store persist directory is required")
        
        # Check numeric values
        if self._config.embedding.text_chunk_size <= 0:
            errors.append("Text chunk size must be positive")
        
        if self._config.agent.sql_confidence_threshold < 0 or self._config.agent.sql_confidence_threshold > 1:
            errors.append("SQL confidence threshold must be between 0 and 1")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> SystemConfig:
    """Get the global configuration"""
    return config_manager.config

def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    return True

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully:")
    print(f"  - Embedding model: {config.embedding.text_model}")
    print(f"  - LLM model: {config.llm.model_name}")
    print(f"  - Vector store: {config.vector_store.persist_directory}")
    print(f"  - Debug mode: {config.debug}")
    
    # Validate environment
    if validate_environment():
        print("✅ Environment validation passed")
    else:
        print("❌ Environment validation failed")
    
    # Validate configuration
    if config_manager.validate_config():
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
