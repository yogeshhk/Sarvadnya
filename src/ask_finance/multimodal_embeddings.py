"""
Multi-modal embedding system for different content types
Handles text embeddings, table embeddings, and image descriptions
"""

import asyncio
import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from dataclasses import asdict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
from PIL import Image
import io
import base64

from document_parser import Chunk, TextChunk, TableChunk, ImageChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    """Base class for different embedding strategies"""
    
    @abstractmethod
    async def embed(self, content: Any) -> List[float]:
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass

class TextEmbedder(BaseEmbedder):
    """Text embedding using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim

class TableEmbedder(BaseEmbedder):
    """Table embedding by converting to text representation"""
    
    def __init__(self, text_embedder: TextEmbedder):
        self.text_embedder = text_embedder
    
    async def embed(self, table_content: Dict) -> List[float]:
        """Convert table to text and embed"""
        # Create text representation of table
        text_repr = self._table_to_text(table_content)
        return await self.text_embedder.embed(text_repr)
    
    def get_embedding_dim(self) -> int:
        return self.text_embedder.get_embedding_dim()
    
    def _table_to_text(self, table_content: Dict) -> str:
        """Convert table structure to searchable text"""
        parts = []
        
        # Add column names
        columns = table_content.get('columns', [])
        if columns:
            parts.append(f"Table columns: {', '.join(columns)}")
        
        # Add schema info
        if 'sql_schema' in table_content:
            parts.append(f"Schema: {table_content['sql_schema']}")
        
        # Add sample data (first few rows)
        data = table_content.get('data', [])
        if data:
            sample_size = min(3, len(data))
            parts.append("Sample data:")
            for i in range(sample_size):
                row_text = ", ".join([f"{k}: {v}" for k, v in data[i].items()])
                parts.append(row_text)
        
        # Add summary statistics if available
        shape = table_content.get('shape', (0, 0))
        parts.append(f"Table size: {shape[0]} rows, {shape[1]} columns")
        
        return " | ".join(parts)

class ImageEmbedder(BaseEmbedder):
    """Image embedding using vision-language models"""
    
    def __init__(self, openai_api_key: Optional[str] = None, text_embedder: Optional[TextEmbedder] = None):
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_vision = True
        else:
            self.use_vision = False
            logger.warning("No OpenAI API key provided. Using text embedder for image descriptions.")
        
        self.text_embedder = text_embedder or TextEmbedder()
    
    async def embed(self, image_content: Dict) -> List[float]:
        """Generate embedding for image"""
        if self.use_vision:
            # Use GPT-4V to describe the image then embed the description
            description = await self._describe_image(image_content['image_base64'])
        else:
            # Use provided description or default
            description = image_content.get('description', "Financial chart or diagram")
        
        # Embed the description
        return await self.text_embedder.embed(description)
    
    def get_embedding_dim(self) -> int:
        return self.text_embedder.get_embedding_dim()
    
    async def _describe_image(self, image_b64: str) -> str:
        """Use GPT-4V to describe image content"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this financial chart or diagram in detail, focusing on key data points, trends, and insights that would be useful for analysis and search."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Could not describe image with GPT-4V: {e}")
            return "Financial chart or diagram - description unavailable"

class MultiModalVectorStore:
    """Vector store that handles different content types"""
    
    def __init__(self, collection_name: str = "financial_docs", 
                 persist_directory: str = "./chroma_db"):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.client.get_collection(name=collection_name)
        
        # Initialize embedders
        self.text_embedder = TextEmbedder()
        self.table_embedder = TableEmbedder(self.text_embedder)
        self.image_embedder = ImageEmbedder(text_embedder=self.text_embedder)
        
        # SQL database for structured queries on tables
        self.sql_db_path = "financial_tables.db"
        self._init_sql_db()
    
    def _init_sql_db(self):
        """Initialize SQLite database for table queries"""
        self.sql_conn = sqlite3.connect(self.sql_db_path, check_same_thread=False)
    
    async def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to vector store with appropriate embeddings"""
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Generate embedding based on content type
            if isinstance(chunk, TextChunk):
                embedding = await self.text_embedder.embed(chunk.content)
                document = chunk.content
            elif isinstance(chunk, TableChunk):
                embedding = await self.table_embedder.embed(chunk.content)
                document = self.table_embedder._table_to_text(chunk.content)
                # Also add to SQL database
                self._add_table_to_sql(chunk)
            elif isinstance(chunk, ImageChunk):
                embedding = await self.image_embedder.embed(chunk.content)
                document = chunk.content.get('description', 'Image content')
            else:
                # Default text embedding
                embedding = await self.text_embedder.embed(str(chunk.content))
                document = str(chunk.content)
            
            embeddings.append(embedding)
            documents.append(document)
            metadatas.append(chunk.metadata)
            ids.append(chunk.chunk_id)
            
            # Store embedding in chunk
            chunk.embedding = embedding
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks to vector store")
    
    def _add_table_to_sql(self, table_chunk: TableChunk):
        """Add table data to SQL database"""
        try:
            table_name = f"table_{table_chunk.chunk_id.replace('-', '_')}"
            
            # Create DataFrame from stored data
            import pandas as pd
            df = pd.DataFrame(table_chunk.content['data'])
            
            # Write to SQLite
            df.to_sql(table_name, self.sql_conn, if_exists='replace', index=False)
            
            # Store table name in metadata
            table_chunk.metadata['sql_table_name'] = table_name
            
            logger.info(f"Added table {table_name} to SQL database")
        except Exception as e:
            logger.warning(f"Could not add table to SQL database: {e}")
    
    async def search(self, query: str, content_types: Optional[List[str]] = None,
                    n_results: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        # Generate query embedding
        query_embedding = await self.text_embedder.embed(query)
        
        # Build where clause for content types
        where_clause = {}
        if content_types:
            where_clause["content_type"] = {"$in": content_types}
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'chunk_id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def execute_sql_query(self, sql_query: str, table_name: str) -> List[Dict]:
        """Execute SQL query on table data"""
        try:
            cursor = self.sql_conn.cursor()
            # Replace placeholder with actual table name
            actual_query = sql_query.replace('TABLE_NAME', table_name)
            cursor.execute(actual_query)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dicts
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return []
    
    def get_table_schema(self, table_name: str) -> Optional[str]:
        """Get schema for a table"""
        try:
            cursor = self.sql_conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema_info = cursor.fetchall()
            
            schema_parts = []
            for col_info in schema_info:
                col_name = col_info[1]
                col_type = col_info[2]
                schema_parts.append(f"{col_name} {col_type}")
            
            return f"CREATE TABLE {table_name} ({', '.join(schema_parts)})"
        except Exception as e:
            logger.error(f"Could not get schema for {table_name}: {e}")
            return None
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve specific chunk by ID"""
        try:
            result = self.collection.get(ids=[chunk_id])
            if result['ids']:
                return {
                    'chunk_id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            logger.error(f"Could not retrieve chunk {chunk_id}: {e}")
        return None

async def main():
    """Example usage"""
    from document_parser import load_chunks
    
    # Load processed chunks
    chunks = load_chunks("financial_chunks.json")
    
    # Initialize vector store
    vector_store = MultiModalVectorStore()
    
    # Add chunks to vector store
    await vector_store.add_chunks(chunks)
    
    # Example searches
    queries = [
        "revenue growth trends",
        "financial ratios and metrics", 
        "cash flow statements",
        "market performance charts"
    ]
    
    for query in queries:
        print(f"\nSearching for: {query}")
        results = await vector_store.search(query, n_results=3)
        
        for result in results:
            print(f"- {result['metadata']['content_source']}: {result['content'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
