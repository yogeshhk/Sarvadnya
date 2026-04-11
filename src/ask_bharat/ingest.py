from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

loader = PyPDFLoader("data/AncientIndia_TwitterThreads_ChinmayBhandari_cleaned.pdf")
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# SemanticChunker groups sentences by embedding similarity, placing split points where
# the cosine distance between adjacent sentence embeddings exceeds the 95th-percentile
# threshold. This finds natural topic boundaries rather than splitting mid-sentence at a
# fixed character count, producing more coherent chunks for retrieval.
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",   # split where similarity drops sharply
    breakpoint_threshold_amount=95,           # top 5% of similarity drops become splits
)
docs_split = splitter.split_documents(docs)

print(f"Split into {len(docs_split)} semantic chunks")

db = FAISS.from_documents(docs_split, embeddings)
db.save_local("vectorstore/db_faiss")
print("✅ Vectorstore saved!")
