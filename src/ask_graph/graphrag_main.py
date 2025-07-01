import os
import sys
import json
import re
import traceback
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class GraphRAG:
    def __init__(self, groq_api_key: str, chroma_persist_directory: str = "./chroma_db"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.graph = nx.Graph()
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name="graph_rag_nodes")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.node_texts = {}
        self.node_embeddings = {}

    def load_and_segment_data(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            df = pd.read_excel(file_path)
            segments = []
            for idx, row in df.iterrows():
                text_content = " ".join([str(val) for val in row.values if pd.notna(val)])
                segments.append({
                    'id': f"node_{idx}",
                    'text': text_content,
                    'metadata': row.to_dict()
                })
            return segments
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return []

    def create_dummy_graph(self, segments: List[Dict[str, Any]]) -> None:
        for segment in segments:
            node_id = segment['id']
            text = segment['text']
            embedding = self.embedding_model.encode(text)
            self.graph.add_node(node_id, text=text, metadata=segment['metadata'])
            self.node_texts[node_id] = text
            self.node_embeddings[node_id] = embedding

        nodes = list(self.graph.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                emb1 = self.node_embeddings[node1]
                emb2 = self.node_embeddings[node2]
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if similarity > 0.3:
                    self.graph.add_edge(node1, node2, weight=similarity)

    def store_in_chroma(self) -> None:
        ids = list(self.node_texts.keys())
        texts = list(self.node_texts.values())
        embeddings = [self.node_embeddings[node_id].tolist() for node_id in ids]

        try:
            existing = self.collection.get()
            existing_ids = existing.get("ids", [])
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        except Exception as e:
            print(f"⚠️ Warning while deleting existing entries: {e}")

    def retrieve_by_embedding_similarity(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query)
        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        return [
            {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'distance': results['distances'][0][i]
            }
            for i in range(len(results['ids'][0]))
        ]

    def get_graph_neighbors(self, node_ids: List[str], depth: int = 1) -> List[str]:
        all_neighbors = set()
        for node_id in node_ids:
            if node_id in self.graph:
                neighbors = nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth)
                all_neighbors.update(neighbors.keys())
        return list(all_neighbors)

    def query_to_graph_query(self, query: str) -> List[str]:
        prompt = f"""
        Convert the following natural language query into a graph query format.
        Extract key entities and relationships that should be searched in a knowledge graph.
        Query: "{query}"
        Return a list of node types or entities that should be retrieved from the graph.
        Format: ["entity1", "entity2"]
        """
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1
            )
            response_text = response.choices[0].message.content
            try:
                entities = json.loads(response_text)
                return entities if isinstance(entities, list) else []
            except:
                return re.findall(r'[\"\']([^\"\']+)[\"\']', response_text) or [query]
        except Exception as e:
            print(f"❌ Error in query conversion: {e}")
            return [query]

    def retrieve_by_graph_query(self, entities: List[str], top_k: int = 5) -> List[str]:
        matched_nodes = set()
        for entity in entities:
            for node_id, data in self.graph.nodes(data=True):
                if entity.lower() in data.get('text', '').lower():
                    matched_nodes.add(node_id)
        return list(matched_nodes)[:top_k]

    def select_best_context(self, query: str, retrieved_nodes: List[Dict[str, Any]]) -> str:
        context_texts = [f"Node {node['id']}: {node['text']}" for node in retrieved_nodes]
        combined_context = "\n\n".join(context_texts)
        prompt = f"""
        Based on the query and retrieved context, summarize the most relevant information.

        Query: {query}
        Context:
        {combined_context}
        """
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error in context selection: {e}")
            return combined_context

    def hybrid_retrieve(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        similar_nodes = self.retrieve_by_embedding_similarity(query, top_k)
        entities = self.query_to_graph_query(query)
        graph_nodes = self.retrieve_by_graph_query(entities, top_k)
        all_ids = [node['id'] for node in similar_nodes] + graph_nodes
        neighbor_ids = self.get_graph_neighbors(all_ids, depth=1)

        final_nodes = {node['id']: node for node in similar_nodes}
        for node_id in graph_nodes + neighbor_ids:
            if node_id not in final_nodes and node_id in self.node_texts:
                final_nodes[node_id] = {
                    'id': node_id,
                    'text': self.node_texts[node_id],
                    'distance': 0.5
                }

        retrieved_list = list(final_nodes.values())[:top_k]
        best_context = self.select_best_context(query, retrieved_list)
        return best_context, retrieved_list

    def answer_query(self, query: str) -> Dict[str, Any]:
        context, nodes = self.hybrid_retrieve(query)
        prompt = f"""
        Based on the context below, answer the user's query.

        Query: {query}
        Context: {context}
        """
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.3
            )
            return {
                'answer': response.choices[0].message.content,
                'context': context,
                'retrieved_nodes': nodes,
                'num_nodes_retrieved': len(nodes)
            }
        except Exception as e:
            return {
                'answer': f"❌ Error generating answer: {e}",
                'context': context,
                'retrieved_nodes': nodes,
                'num_nodes_retrieved': len(nodes)
            }


def create_sample_data() -> str:
    data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
        'Category': ['Electronics'] * 5,
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
    file_path = 'sample_data.xlsx'
    df.to_excel(file_path, index=False)
    return file_path


if __name__ == "__main__":
    print("🚀 Testing GraphRAG system...")
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env file.")
        sys.exit(1)

    try:
        graph_rag = GraphRAG(GROQ_API_KEY)
        sample_file = create_sample_data()
        print(f"📄 Sample data file created: {sample_file}")
        segments = graph_rag.load_and_segment_data(sample_file)
        print(f"📊 Loaded {len(segments)} segments.")
        graph_rag.create_dummy_graph(segments)
        print(f"📈 Graph has {graph_rag.graph.number_of_nodes()} nodes and {graph_rag.graph.number_of_edges()} edges.")
        graph_rag.store_in_chroma()
        print("💾 Stored embeddings in ChromaDB.")

        queries = [
            "What laptops are available?",
            "Show me electronics under $100",
            "What are the monitor specifications?"
        ]

        for query in queries:
            print(f"\n🔍 Query: {query}")
            result = graph_rag.answer_query(query)
            print(f"📘 Answer: {result['answer']}")
            print(f"🧠 Retrieved Nodes: {result['num_nodes_retrieved']}")

        print("\n✅ Test completed successfully.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")
