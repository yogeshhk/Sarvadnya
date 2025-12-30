import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from chromadb import PersistentClient

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if not hf_token:
    raise ValueError("❌ Hugging Face token not found in .env file. Please add HUGGINGFACE_TOKEN to your .env.")

class RAGChatbot:
    def __init__(self, data_directory: str):
        print("🧠 Loading documents...")
        documents = SimpleDirectoryReader(data_directory).load_data()

        print("🔍 Splitting into chunks...")
        parser = SentenceSplitter(chunk_size=256, chunk_overlap=30)

        print("📌 Setting up embedding model (MURIL)...")
        embed_model = HuggingFaceEmbedding(
            model_name="google/muril-base-cased",
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=512
        )

        Settings.embed_model = embed_model
        Settings.node_parser = parser
        Settings.llm = MockLLM()

        print("📚 Setting up Chroma vector store...")
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)

        try:
            chroma_client = PersistentClient(path=persist_directory)
            chroma_client.delete_collection("muril_docs")
        except Exception as e:
            print(f"⚠️ Collection deletion warning: {e}")

        chroma_client = PersistentClient(path=persist_directory)
        chroma_collection = chroma_client.get_or_create_collection("muril_docs")

        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection,
            client=chroma_client
        )

        print("⚙️ Creating vector index...")
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, vector_store=vector_store)

        retriever = index.as_retriever(similarity_top_k=3)
        retriever.similarity_cutoff = 0.65

        self.query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
        print("✅ RAG setup complete.")

    def extract_answer_from_context(self, context_nodes, query):
        if not context_nodes:
            return "❌ संदर्भ सापडले नाहीत."

        print("\n🧠 Retrieved Nodes:")
        for i, node in enumerate(context_nodes):
            print(f"\n🔹 Node {i+1} (Score: {node.score:.3f}):")
            print(node.text.strip()[:300] + "...\n")

        best_node = max(context_nodes, key=lambda x: x.score)
        answer = best_node.text.strip()

        for line in answer.split("\n"):
            if any(word in line for word in query.split()):
                return line.strip()

        return answer if answer else "माफ करा, मला या प्रश्नाचे उत्तर सापडले नाही."

    def chat(self, query: str) -> str:
        print(f"\n🧾 User: {query}")
        response = self.query_engine.query(query)
        context_nodes = response.source_nodes
        answer = self.extract_answer_from_context(context_nodes, query)
        return answer

if __name__ == "__main__":
    print("🛁 Done with your shower? Let's test the chatbot.")
    bot = RAGChatbot(data_directory="data")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = bot.chat(user_input)
        print("🤖 Shivneri Bot:", response)
