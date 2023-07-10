# https://github.com/kylesteckler/generative-ai/blob/main/notebooks/knowledge_based_system.ipynb

from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

docs = WikipediaLoader(query="Machine Learning", load_max_docs=10).load()
docs += WikipediaLoader(query="Deep Learning", load_max_docs=10).load()
docs += WikipediaLoader(query="Neural Networks", load_max_docs=10).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
)

chunks = text_splitter.split_documents(docs)

# Look at the first two chunks
print(chunks[0:2])
print(f'Number of documents: {len(docs)}')
print(f'Number of chunks: {len(chunks)}')

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


ask_question("What is a gradient boosted tree?")

ask_question("When was the transformer invented?")

ask_question("What technology underpins large language models?")

# preserve chat history in memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_session = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

chat_session({'question': 'What technology underpins large language models?'})

# With chat history it will understand that "they" refers to transformers
chat_session({'question': 'When were they invented?'})