import os
import pandas as pd
import networkx as nx
import chromadb
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from groq import Groq
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import json
import re

class GraphRAG:
    def __init__(self, groq_api_key: str, chroma_persist_directory: str = "./chroma_db"):
        """Initialize GraphRAG with necessary components"""
        self.groq_client = Groq(api_key=groq_api_key)
        self.graph = nx.Graph()
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name="graph_rag_nodes")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LlamaIndex embedding
        self.llama_embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Node storage
        self.node_texts = {}
        self.node_embeddings = {}
        
    def load_and_segment_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from Excel file and segment into nodes"""
        try:
            df = pd.read_excel(file_path)
            segments = []
            
            for idx, row in df.iterrows():
                # Create segments from each row
                text_content = " ".join([str(val) for val in row.values if pd.notna(val)])
                segment = {
                    'id': f"node_{idx}",
                    'text': text_content,
                    'metadata': row.to_dict()
                }
                segments.append(segment)
            
            return segments
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def create_dummy_graph(self, segments: List[Dict[str, Any]]) -> None:
        """Create dummy graph with nodes and edges"""
        # Add nodes to graph
        for segment in segments:
            node_id = segment['id']
            text = segment['text']
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Store in graph
            self.graph.add_node(node_id, text=text, metadata=segment['metadata'])
            
            # Store embeddings and text
            self.node_texts[node_id] = text
            self.node_embeddings[node_id] = embedding
        
        # Create edges based on text similarity
        nodes = list(self.graph.nodes())
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # Calculate similarity
                emb1 = self.node_embeddings[node1]
                emb2 = self.node_embeddings[node2]
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                # Add edge if similarity is above threshold
                if similarity > 0.3:
                    self.graph.add_edge(node1, node2, weight=similarity)
    
    def store_in_chroma(self) -> None:
        """Store node embeddings in ChromaDB"""
        ids = list(self.node_texts.keys())
        texts = list(self.node_texts.values())
        embeddings = [self.node_embeddings[node_id].tolist() for node_id in ids]
        
        # Clear existing collection
        self.collection.delete(where={})
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings
        )
    
    def retrieve_by_embedding_similarity(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve nodes by embedding similarity"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        retrieved_nodes = []
        for i, node_id in enumerate(results['ids'][0]):
            retrieved_nodes.append({
                'id': node_id,
                'text': results['documents'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_nodes
    
    def get_graph_neighbors(self, node_ids: List[str], depth: int = 1) -> List[str]:
        """Get neighbors from graph"""
        all_neighbors = set()
        
        for node_id in node_ids:
            if node_id in self.graph:
                # Get neighbors at specified depth
                neighbors = nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth)
                all_neighbors.update(neighbors.keys())
        
        return list(all_neighbors)
    
    def query_to_graph_query(self, query: str) -> List[str]:
        """Convert English query to graph query language using LLM"""
        prompt = f"""
        Convert the following natural language query into a graph query format.
        Extract key entities and relationships that should be searched in a knowledge graph.
        
        Query: "{query}"
        
        Return a list of node types or entities that should be retrieved from the graph.
        Format your response as a JSON list of strings.
        
        Example: ["entity1", "entity2", "relationship_type"]
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            import json
            try:
                entities = json.loads(response_text)
                return entities if isinstance(entities, list) else []
            except:
                # Fallback: extract words in quotes or brackets
                matches = re.findall(r'[\"\']([^\"\']+)[\"\']', response_text)
                return matches if matches else [query]
                
        except Exception as e:
            print(f"Error in query conversion: {e}")
            return [query]
    
    def retrieve_by_graph_query(self, entities: List[str], top_k: int = 5) -> List[str]:
        """Retrieve nodes based on graph query entities"""
        relevant_nodes = []
        
        for entity in entities:
            # Find nodes containing the entity
            for node_id, data in self.graph.nodes(data=True):
                text = data.get('text', '').lower()
                if entity.lower() in text:
                    relevant_nodes.append(node_id)
        
        # Remove duplicates and limit results
        return list(set(relevant_nodes))[:top_k]
    
    def select_best_context(self, query: str, retrieved_nodes: List[Dict[str, Any]]) -> str:
        """Use LLM to select best information for context"""
        context_texts = []
        for node in retrieved_nodes:
            context_texts.append(f"Node {node['id']}: {node['text']}")
        
        combined_context = "\n\n".join(context_texts)
        
        prompt = f"""
        Given the following query and retrieved information, select and summarize the most relevant context.
        
        Query: "{query}"
        
        Retrieved Information:
        {combined_context}
        
        Provide a concise summary of the most relevant information that would help answer the query.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in context selection: {e}")
            return combined_context
    
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """Hybrid retrieval combining embedding similarity and graph traversal"""
        # Method 1: Embedding similarity
        similar_nodes = self.retrieve_by_embedding_similarity(query, top_k)
        
        # Method 2: Graph query approach
        entities = self.query_to_graph_query(query)
        graph_nodes = self.retrieve_by_graph_query(entities, top_k)
        
        # Get graph neighbors
        all_node_ids = [node['id'] for node in similar_nodes] + graph_nodes
        neighbor_ids = self.get_graph_neighbors(all_node_ids, depth=1)
        
        # Combine all retrieved nodes
        all_retrieved = {}
        
        # Add similarity-based nodes
        for node in similar_nodes:
            all_retrieved[node['id']] = node
        
        # Add graph-based nodes
        for node_id in graph_nodes + neighbor_ids:
            if node_id not in all_retrieved and node_id in self.node_texts:
                all_retrieved[node_id] = {
                    'id': node_id,
                    'text': self.node_texts[node_id],
                    'distance': 0.5  # Default distance for graph-retrieved nodes
                }
        
        retrieved_list = list(all_retrieved.values())[:top_k]
        
        # Select best context
        best_context = self.select_best_context(query, retrieved_list)
        
        return best_context, retrieved_list
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Main function to answer query using GraphRAG"""
        context, retrieved_nodes = self.hybrid_retrieve(query)
        
        # Generate final answer
        prompt = f"""
        Based on the following context, answer the user's query.
        
        Context: {context}
        
        Query: {query}
        
        Provide a comprehensive answer based on the available information.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'context': context,
                'retrieved_nodes': retrieved_nodes,
                'num_nodes_retrieved': len(retrieved_nodes)
            }
        except Exception as e:
            return {
                'answer': f"Error generating answer: {e}",
                'context': context,
                'retrieved_nodes': retrieved_nodes,
                'num_nodes_retrieved': len(retrieved_nodes)
            }

# Test functions
def create_sample_data():
    """Create sample Excel data for testing"""
    data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
        'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'Price': [999.99, 29.99, 79.99, 299.99, 89.99],
        'Description': [
            'High-performance laptop with 16GB RAM and SSD storage',
            'Wireless optical mouse with ergonomic design',
            'Mechanical keyboard with RGB backlighting',
            '27-inch 4K monitor with HDR support',
            '1080p webcam with built-in microphone'
        ],
        'Brand': ['TechCorp', 'MouseMaster', 'KeyPro', 'ViewTech', 'CamMax']
    }
    
    df = pd.DataFrame(data)
    df.to_excel('sample_data.xlsx', index=False)
    return 'sample_data.xlsx'

if __name__ == "__main__":
    # Test the GraphRAG system
    print("Testing GraphRAG system...")
    
    # Set your Groq API key
    GROQ_API_KEY = "your_groq_api_key_here"  # Replace with actual key
    
    try:
        # Initialize GraphRAG
        graph_rag = GraphRAG(GROQ_API_KEY)
        
        # Create sample data
        sample_file = create_sample_data()
        print(f"Created sample data: {sample_file}")
        
        # Load and segment data
        segments = graph_rag.load_and_segment_data(sample_file)
        print(f"Loaded {len(segments)} segments")
        
        # Create graph
        graph_rag.create_dummy_graph(segments)
        print(f"Created graph with {graph_rag.graph.number_of_nodes()} nodes and {graph_rag.graph.number_of_edges()} edges")
        
        # Store in ChromaDB
        graph_rag.store_in_chroma()
        print("Stored embeddings in ChromaDB")
        
        # Test queries
        test_queries = [
            "What laptops are available?",
            "Show me electronics under $100",
            "What are the monitor specifications?"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing Query: '{query}' ---")
            result = graph_rag.answer_query(query)
            print(f"Answer: {result['answer']}")
            print(f"Retrieved {result['num_nodes_retrieved']} nodes")
            
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure to replace 'your_groq_api_key_here' with your actual Groq API key")
