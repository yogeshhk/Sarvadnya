from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Use GROQ API as OpenAI-compatible LLM
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# Load Wikipedia topics
topics = ["Machine Learning", "Deep Learning", "Neural Networks"]
docs = []
for topic in topics:
    docs += WikipediaLoader(query=topic, load_max_docs=10).load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)
chunks = text_splitter.split_documents(docs)
print(f" Loaded {len(docs)} documents and split into {len(chunks)} chunks.")

# Use HuggingFace embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector DB and retriever
db = Chroma.from_documents(chunks, embedding, persist_directory="./vectorstore")
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10})  # k=2 for 2 relevant unique results

# Load Groq LLM using ChatOpenAI wrapper
llm = ChatOpenAI(
    model="llama3-70b-8192",
    temperature=0.7,
    max_tokens=1024
)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Function to ask questions and remove duplicate links
def ask_question(question: str):
    response = qa.invoke({"query": question})
    print(f"\n Question: {question}")
    print(f" Answer: {response['result']}\n")

    # Show only 2 unique source links
    unique_links = []
    seen = set()
    for doc in response["source_documents"]:
        src = doc.metadata.get("source", "Wikipedia")
        if src not in seen:
            seen.add(src)
            unique_links.append(src)
        if len(unique_links) == 2:
            break

    print(" Sources:")
    for link in unique_links:
        print(f"- {link}")

# Ask questions
# ask_question("What is a gradient boosted tree?")
# ask_question("When was the transformer invented?")
# ask_question("What technology underpins large language models?")
ask_question("What is multimodal learning?")
ask_question("What is few-shot and zero-shot learning?")
ask_question("What is a neural network?")

