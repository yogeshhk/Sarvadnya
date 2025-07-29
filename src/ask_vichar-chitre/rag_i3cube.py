import os
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser

# ✅ Load environment variables
load_dotenv()
print("[INFO] LLM is disabled. Using only vector search from documents.")
Settings.llm = None  # Disable LLM completely

# ✅ Use L3Cube Marathi Sentence-BERT for embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="l3cube-pune/marathi-sentence-bert-nli"
)

# ✅ Main RAG class
class RAGL3Cube:
    def __init__(self, data_directory="./data"):
        print(f"📂 Loading documents from: {data_directory}")
        self.documents = SimpleDirectoryReader(
            data_directory,
            recursive=True,
            filename_as_id=True,
            required_exts=[".tex"]
        ).load_data()
        print(f"✅ Loaded {len(self.documents)} documents.")

        # Split documents into small chunks for concise retrieval
        parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(self.documents)

        # Create vector store index
        self.index = VectorStoreIndex(nodes)

        # Query engine with no LLM (only retrieval)
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,   # Top 3 relevant chunks
            llm=None,
            response_mode="no_text"  # Return only retrieved chunks
        )

    def ask(self, question: str) -> str:
        response = self.query_engine.query(question)

        if not response.source_nodes:
            return "माफ करा, संबंधित माहिती सापडली नाही."

        results = []
        for i, node in enumerate(response.source_nodes, start=1):
            score = round(node.score or 0, 3)
            text = node.node.text.strip().replace("\n", " ")
            
            # Limit to first 200 chars for concise output
            short_text = text[:200] + "..." if len(text) > 200 else text
            results.append(f"🔹 [Score: {score}] {short_text}")

        return "\n".join(results)


# ✅ CLI Chat Loop
if __name__ == "__main__":
    bot = RAGL3Cube(data_directory="./data")
    while True:
        try:
            question = input("\n❓ तुमचा प्रश्न विचाराः ")
            if question.strip().lower() in ["exit", "बाहेर पडा"]:
                print("👋 बाय! बाहेर पडत आहे... ✅")
                break
            answer = bot.ask(question)
            print("\n✅ उत्तर:\n", answer)
        except Exception as e:
            print("❌ एरर:", str(e))
