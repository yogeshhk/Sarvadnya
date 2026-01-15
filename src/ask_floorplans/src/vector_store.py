"""
Vector Store Implementation for Floor Plan Embeddings
Supports FAISS (local) and Pinecone (production)
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class VectorStore(ABC):
    """Abstract base class for vector storage"""
    
    @abstractmethod
    def add_embedding(self, plan_id: str, embedding: np.ndarray, metadata: Dict) -> None:
        """Add an embedding with metadata"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar embeddings"""
        pass
    
    @abstractmethod
    def delete(self, plan_id: str) -> bool:
        """Delete an embedding by ID"""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for local development"""
    
    def __init__(self, dimension: int = 384, index_path: str = "faiss.index"):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_path: Path to save/load FAISS index
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = index_path.replace('.index', '_metadata.json')
        
        # Initialize FAISS index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata_store = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata_store = {}
        
        self.id_to_index = {}
        self.index_to_id = {}
    
    def add_embedding(
        self, 
        plan_id: str, 
        embedding: np.ndarray, 
        metadata: Dict
    ) -> None:
        """Add embedding to FAISS index"""
        # Ensure embedding is the right shape
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Add to index
        current_index = self.index.ntotal
        self.index.add(embedding.astype('float32'))
        
        # Store mappings
        self.id_to_index[plan_id] = current_index
        self.index_to_id[current_index] = plan_id
        
        # Store metadata
        self.metadata_store[plan_id] = metadata
        
        # Save periodically
        self._save()
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar embeddings"""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k * 2, self.index.ntotal)  # Get more for filtering
        )
        
        # Retrieve results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            plan_id = self.index_to_id.get(idx)
            if plan_id is None:
                continue
            
            metadata = self.metadata_store.get(plan_id, {})
            
            # Apply filters if provided
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            results.append({
                'id': plan_id,
                'score': float(dist),
                'metadata': metadata
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def delete(self, plan_id: str) -> bool:
        """Delete embedding (mark as deleted in metadata)"""
        if plan_id in self.metadata_store:
            del self.metadata_store[plan_id]
            self._save()
            return True
        return False
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            # Handle different filter types
            if isinstance(value, dict):
                # Range filters: {"area": {"$gte": 100, "$lte": 1000}}
                meta_value = metadata[key]
                if '$gte' in value and meta_value < value['$gte']:
                    return False
                if '$lte' in value and meta_value > value['$lte']:
                    return False
                if '$gt' in value and meta_value <= value['$gt']:
                    return False
                if '$lt' in value and meta_value < value['$lt']:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False
        
        return True
    
    def _save(self) -> None:
        """Save index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata_store, f, indent=2)


class PineconeVectorStore(VectorStore):
    """Pinecone-based vector store for production"""
    
    def __init__(
        self, 
        index_name: str = "floor-plans",
        dimension: int = 384
    ):
        """Initialize Pinecone vector store"""
        try:
            import pinecone
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError(
                "Pinecone not installed. Install with: pip install pinecone-client"
            )
        
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set in environment")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        self.index = pc.Index(index_name)
    
    def add_embedding(
        self, 
        plan_id: str, 
        embedding: np.ndarray, 
        metadata: Dict
    ) -> None:
        """Add embedding to Pinecone"""
        self.index.upsert(
            vectors=[(plan_id, embedding.tolist(), metadata)]
        )
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search in Pinecone"""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True,
            filter=filters
        )
        
        return [
            {
                'id': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            }
            for match in results['matches']
        ]
    
    def delete(self, plan_id: str) -> bool:
        """Delete from Pinecone"""
        self.index.delete(ids=[plan_id])
        return True


class EmbeddingGenerator:
    """Generate embeddings for floor plans"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def generate_plan_embedding(self, floor_plan: Dict) -> np.ndarray:
        """
        Generate embedding for a floor plan
        
        Args:
            floor_plan: Floor plan dict
            
        Returns:
            Embedding vector
        """
        # Create text representation
        text = self._plan_to_text(floor_plan)
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        return self.model.encode(query, convert_to_numpy=True)
    
    def _plan_to_text(self, floor_plan: Dict) -> str:
        """Convert floor plan to searchable text"""
        parts = []
        
        # Basic info
        parts.append(f"Floor plan: {floor_plan.get('name', 'Unnamed')}")
        parts.append(f"Total area: {floor_plan.get('total_area', 0)} square meters")
        
        # Count rooms by type
        rooms = floor_plan.get('rooms', [])
        room_types = {}
        for room in rooms:
            room_type = room.get('type', 'unknown')
            room_types[room_type] = room_types.get(room_type, 0) + 1
        
        for room_type, count in room_types.items():
            parts.append(f"{count} {room_type}{'s' if count > 1 else ''}")
        
        # Features
        has_balcony = any(
            room.get('features', {}).get('balcony', False) 
            for room in rooms
        )
        if has_balcony:
            parts.append("with balcony")
        
        # Zones
        zones = floor_plan.get('zones', [])
        if zones:
            zone_names = [z.get('name', '') for z in zones]
            parts.append(f"zones: {', '.join(zone_names)}")
        
        # Design notes
        constraints = floor_plan.get('constraints', {})
        notes = constraints.get('design_notes', '')
        if notes:
            parts.append(f"design: {notes}")
        
        return " | ".join(parts)


class FloorPlanVectorSearch:
    """High-level interface for floor plan vector search"""
    
    def __init__(
        self, 
        store_type: str = "faiss",
        index_name: str = "floor-plans"
    ):
        """
        Initialize vector search system
        
        Args:
            store_type: 'faiss' or 'pinecone'
            index_name: Index name
        """
        self.embedding_generator = EmbeddingGenerator()
        
        if store_type == "faiss":
            self.store = FAISSVectorStore(
                dimension=self.embedding_generator.dimension,
                index_path=f"{index_name}.index"
            )
        elif store_type == "pinecone":
            self.store = PineconeVectorStore(
                index_name=index_name,
                dimension=self.embedding_generator.dimension
            )
        else:
            raise ValueError(f"Unknown store type: {store_type}")
    
    def index_floor_plan(self, floor_plan: Dict) -> None:
        """Index a floor plan for search"""
        plan_id = floor_plan.get('id')
        if not plan_id:
            raise ValueError("Floor plan must have an 'id' field")
        
        # Generate embedding
        embedding = self.embedding_generator.generate_plan_embedding(floor_plan)
        
        # Prepare metadata for filtering
        metadata = {
            'name': floor_plan.get('name', ''),
            'total_area': floor_plan.get('total_area', 0),
            'level': floor_plan.get('level', 0),
        }
        
        # Add room counts
        rooms = floor_plan.get('rooms', [])
        metadata['bedroom_count'] = sum(
            1 for r in rooms if r.get('type') == 'bedroom'
        )
        metadata['bathroom_count'] = sum(
            1 for r in rooms if r.get('type') == 'bathroom'
        )
        metadata['total_rooms'] = len(rooms)
        
        # Add to store
        self.store.add_embedding(plan_id, embedding, metadata)
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for floor plans
        
        Args:
            query: Natural language query
            k: Number of results
            filters: Additional filters
            
        Returns:
            List of matching floor plans
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search
        results = self.store.search(query_embedding, k=k, filters=filters)
        
        return results


# Example usage
if __name__ == "__main__":
    from floor_plan_schema import EXAMPLE_FLOOR_PLAN
    
    # Initialize search system
    search = FloorPlanVectorSearch(store_type="faiss")
    
    # Index example floor plan
    search.index_floor_plan(EXAMPLE_FLOOR_PLAN)
    
    # Search examples
    print("Search: 2 bedroom apartments")
    results = search.search("2 bedroom apartments", k=5)
    for result in results:
        print(f"  - {result['metadata']['name']} (score: {result['score']:.3f})")
    
    print("\nSearch with filters: apartments under 1000 sqm")
    results = search.search(
        "modern apartments",
        k=5,
        filters={'total_area': {'$lte': 1000}}
    )
    for result in results:
        print(f"  - {result['metadata']['name']} ({result['metadata']['total_area']} sqm)")
