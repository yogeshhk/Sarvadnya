# https://github.com/kylesteckler/generative-ai/blob/main/notebooks/knowledge_based_system.ipynb

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

documents = PyPDFLoader(file_path='../data/On-Paul-Graham-2.pdf').load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
)

chunks = text_splitter.split_documents(documents)

embedding = VertexAIEmbeddings()  # PaLM embedding API

# set persist directory so the vector store is saved to disk
db = Chroma.from_documents(chunks, embedding, persist_directory="./vectorstore")

# vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # number of nearest neighbors to retrieve
)

# PaLM API
# You can also set temperature, top_p, top_k
llm = VertexAI(
    model_name="text-bison",
    max_output_tokens=1024
)

# q/a chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


def ask_question(question: str):
    response = qa({"query": question})
    print(f"Response: {response['result']}\n")

    citations = {doc.metadata['source'] for doc in response['source_documents']}
    print(f"Citations: {citations}\n")

    # uncomment below to print source chunks used
    print(f"Source Chunks Used: {response['source_documents']}")


ask_question("What is the theme of the documents?")
