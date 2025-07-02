import pickle
import faiss
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from config import FAISS_STORE_PKL, DOCS_INDEX

# Load FAISS index
index = faiss.read_index(DOCS_INDEX)
with open(FAISS_STORE_PKL, "rb") as f:
    store = pickle.load(f)
store.index = index

# Load Groq LLM
llm = ChatGroq(
    api_key="GROQ_API_KEY",  # use os.getenv("GROQ_API_KEY")
    model_name="llama3-70b-8192"
)

# Set up the QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever())

# Interactive prompt
while True:
    query = input("Ask something (or 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa.invoke({"query": query})
    print(f"\n Answer: {result['result']}\n")
