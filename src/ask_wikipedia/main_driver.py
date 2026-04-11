from dotenv import load_dotenv
import os

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Load Wikipedia topics
topics = ["Machine Learning", "Deep Learning", "Neural Networks"]
docs = []
for topic in topics:
    docs += WikipediaLoader(query=topic, load_max_docs=10).load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)
chunks = text_splitter.split_documents(docs)
print(f"Loaded {len(docs)} documents and split into {len(chunks)} chunks.")

# HuggingFace embeddings (matches model used in other projects)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build vectorstore; Chroma persists automatically when persist_directory is set
db = Chroma.from_documents(chunks, embedding, persist_directory="./vectorstore")
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10})

# Groq LLM — direct integration, no OpenAI-compatibility wrapper needed
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("GROQ_API_KEY is not set. Export it before running.")

llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-70b-8192",
    temperature=0.7,
    max_tokens=1024,
)


def ask_question(question: str):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    messages = [
        SystemMessage(content=(
            "Answer the question using only the provided context. "
            "Be concise and accurate. If the answer is not in the context, say you don't know."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)
    print(f"\nQuestion: {question}")
    print(f"Answer: {response.content}\n")

    # Show up to 2 unique source links
    unique_links: list[str] = []
    seen: set[str] = set()
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "Wikipedia")
        if src not in seen:
            seen.add(src)
            unique_links.append(src)
        if len(unique_links) == 2:
            break

    print("Sources:")
    for link in unique_links:
        print(f"  - {link}")


# ask_question("What is a gradient boosted tree?")
# ask_question("When was the transformer invented?")
# ask_question("What technology underpins large language models?")
ask_question("What is multimodal learning?")
ask_question("What is few-shot and zero-shot learning?")
ask_question("What is a neural network?")

