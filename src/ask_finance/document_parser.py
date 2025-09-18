"""
Multi-modal Document Parser for Financial Documents
Handles text, tables, images, and charts with specialized chunking strategies
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import uuid

from docling.document_converter import DocumentConverter
from docling.document_converter import ConversionResult
import pandas as pd
from PIL import Image
import base64
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Base chunk class for all content types"""
    chunk_id: str
    content_type: str  # 'text', 'table', 'image', 'mixed'
    content: Union[str, Dict]
    metadata: Dict
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'chunk_id': self.chunk_id,
            'content_type': self.content_type,
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding
        }

@dataclass
class TextChunk(Chunk):
    """Text-specific chunk"""
    def __init__(self, content: str, metadata: Dict):
        super().__init__(
            chunk_id=str(uuid.uuid4()),
            content_type='text',
            content=content,
            metadata=metadata
        )

@dataclass
class TableChunk(Chunk):
    """Table-specific chunk with SQL schema"""
    def __init__(self, table_data: pd.DataFrame, metadata: Dict, sql_schema: Optional[str] = None):
        # Store table as dict for JSON serialization
        table_dict = {
            'data': table_data.to_dict('records'),
            'columns': list(table_data.columns),
            'shape': table_data.shape,
            'sql_schema': sql_schema or self._generate_sql_schema(table_data)
        }
        
        super().__init__(
            chunk_id=str(uuid.uuid4()),
            content_type='table',
            content=table_dict,
            metadata=metadata
        )
    
    @staticmethod
    def _generate_sql_schema(df: pd.DataFrame) -> str:
        """Generate SQL CREATE TABLE statement"""
        schema_parts = []
        for col in df.columns:
            dtype = df[col].dtype
            if 'int' in str(dtype):
                sql_type = 'INTEGER'
            elif 'float' in str(dtype):
                sql_type = 'REAL'
            else:
                sql_type = 'TEXT'
            schema_parts.append(f"{col} {sql_type}")
        
        return f"CREATE TABLE financial_table ({', '.join(schema_parts)})"

@dataclass
class ImageChunk(Chunk):
    """Image-specific chunk with visual description"""
    def __init__(self, image_data: bytes, metadata: Dict, description: Optional[str] = None):
        # Convert image to base64 for storage
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        content_dict = {
            'image_base64': image_b64,
            'description': description or "Financial chart or diagram",
            'format': metadata.get('format', 'unknown')
        }
        
        super().__init__(
            chunk_id=str(uuid.uuid4()),
            content_type='image',
            content=content_dict,
            metadata=metadata
        )

class MultiModalChunker:
    """Handles chunking of different content types with specialized strategies"""
    
    def __init__(self, text_chunk_size: int = 1000, text_overlap: int = 200):
        self.text_chunk_size = text_chunk_size
        self.text_overlap = text_overlap
        self.converter = DocumentConverter()
    
    def parse_document(self, file_path: str) -> ConversionResult:
        """Parse document using docling"""
        logger.info(f"Parsing document: {file_path}")
        result = self.converter.convert(file_path)
        return result
    
    def chunk_text(self, text: str, metadata: Dict) -> List[TextChunk]:
        """Chunk text content with overlap"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.text_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings in the last 100 characters
                last_period = text.rfind('.', start, end)
                last_exclamation = text.rfind('!', start, end)
                last_question = text.rfind('?', start, end)
                
                sentence_end = max(last_period, last_exclamation, last_question)
                if sentence_end > start + self.text_chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    'start_char': start,
                    'end_char': end,
                    'chunk_length': len(chunk_text)
                }
                chunks.append(TextChunk(chunk_text, chunk_metadata))
            
            start = end - self.text_overlap
        
        return chunks
    
    def chunk_table(self, table_data: pd.DataFrame, metadata: Dict) -> TableChunk:
        """Create single chunk for table with metadata"""
        enhanced_metadata = {
            **metadata,
            'table_shape': table_data.shape,
            'column_names': list(table_data.columns),
            'numeric_columns': [col for col in table_data.columns 
                              if pd.api.types.is_numeric_dtype(table_data[col])],
            'text_columns': [col for col in table_data.columns 
                           if pd.api.types.is_string_dtype(table_data[col])]
        }
        
        return TableChunk(table_data, enhanced_metadata)
    
    def chunk_image(self, image_data: bytes, metadata: Dict, 
                   description: Optional[str] = None) -> ImageChunk:
        """Create chunk for image content"""
        # Try to determine image type
        try:
            img = Image.open(io.BytesIO(image_data))
            img_format = img.format.lower() if img.format else 'unknown'
            img_size = img.size
        except Exception as e:
            logger.warning(f"Could not process image: {e}")
            img_format = 'unknown'
            img_size = (0, 0)
        
        enhanced_metadata = {
            **metadata,
            'format': img_format,
            'size': img_size,
            'data_size': len(image_data)
        }
        
        return ImageChunk(image_data, enhanced_metadata, description)
    
    def process_document(self, file_path: str) -> List[Chunk]:
        """Process entire document and return all chunks"""
        logger.info(f"Processing document: {file_path}")
        
        # Parse document
        result = self.converter.convert(file_path)
        
        all_chunks = []
        doc_metadata = {
            'source_file': file_path,
            'file_hash': self._get_file_hash(file_path)
        }
        
        # Process different content types from docling result
        for page_num, page in enumerate(result.document.pages):
            page_metadata = {**doc_metadata, 'page': page_num}
            
            # Process text content
            if hasattr(page, 'text') and page.text:
                text_chunks = self.chunk_text(page.text, {
                    **page_metadata, 
                    'content_source': 'page_text'
                })
                all_chunks.extend(text_chunks)
            
            # Process tables
            if hasattr(page, 'tables'):
                for table_idx, table in enumerate(page.tables):
                    try:
                        # Convert table to DataFrame
                        df = self._table_to_dataframe(table)
                        if not df.empty:
                            table_chunk = self.chunk_table(df, {
                                **page_metadata,
                                'content_source': 'table',
                                'table_index': table_idx
                            })
                            all_chunks.append(table_chunk)
                    except Exception as e:
                        logger.warning(f"Could not process table {table_idx}: {e}")
            
            # Process images/figures
            if hasattr(page, 'figures'):
                for fig_idx, figure in enumerate(page.figures):
                    try:
                        if hasattr(figure, 'image_data'):
                            img_chunk = self.chunk_image(
                                figure.image_data,
                                {
                                    **page_metadata,
                                    'content_source': 'figure',
                                    'figure_index': fig_idx
                                },
                                getattr(figure, 'caption', None)
                            )
                            all_chunks.append(img_chunk)
                    except Exception as e:
                        logger.warning(f"Could not process figure {fig_idx}: {e}")
        
        logger.info(f"Generated {len(all_chunks)} chunks")
        return all_chunks
    
    def _table_to_dataframe(self, table) -> pd.DataFrame:
        """Convert docling table to pandas DataFrame"""
        # This is a simplified implementation - adjust based on docling's table structure
        if hasattr(table, 'data'):
            return pd.DataFrame(table.data)
        elif hasattr(table, 'rows'):
            data = []
            headers = None
            for row_idx, row in enumerate(table.rows):
                if row_idx == 0 and hasattr(row, 'is_header') and row.is_header:
                    headers = [cell.text for cell in row.cells]
                else:
                    data.append([cell.text for cell in row.cells])
            return pd.DataFrame(data, columns=headers)
        return pd.DataFrame()
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

def save_chunks(chunks: List[Chunk], output_path: str):
    """Save chunks to JSON file"""
    chunks_data = [chunk.to_dict() for chunk in chunks]
    with open(output_path, 'w') as f:
        json.dump(chunks_data, f, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")

def load_chunks(input_path: str) -> List[Chunk]:
    """Load chunks from JSON file"""
    with open(input_path, 'r') as f:
        chunks_data = json.load(f)
    
    chunks = []
    for data in chunks_data:
        if data['content_type'] == 'text':
            chunk = TextChunk(data['content'], data['metadata'])
        elif data['content_type'] == 'table':
            # Recreate DataFrame from stored data
            df = pd.DataFrame(data['content']['data'])
            chunk = TableChunk(df, data['metadata'])
        elif data['content_type'] == 'image':
            # Decode base64 image
            image_data = base64.b64decode(data['content']['image_base64'])
            chunk = ImageChunk(image_data, data['metadata'], 
                             data['content'].get('description'))
        else:
            # Generic chunk
            chunk = Chunk(
                data['chunk_id'],
                data['content_type'],
                data['content'],
                data['metadata']
            )
        
        chunk.chunk_id = data['chunk_id']
        chunk.embedding = data.get('embedding')
        chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks

if __name__ == "__main__":
    # Example usage
    chunker = MultiModalChunker(text_chunk_size=800, text_overlap=150)
    
    # Process a financial document
    file_path = "./data/wipo_pub_rn2021_18e.pdf"
    chunks = chunker.process_document(file_path)
    
    # Save chunks
    save_chunks(chunks, "./results/financial_chunks.json")
    
    print(f"Processing complete. Generated {len(chunks)} chunks:")
    for chunk in chunks[:5]:  # Show first 5 chunks
        print(f"- {chunk.content_type}: {chunk.chunk_id}")
