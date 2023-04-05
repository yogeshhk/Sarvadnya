"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from config import *
import os

# Here we load in the data
ps = list(Path(DATA_FOLDER).glob("**/*.txt"))

data = []
sources = []
for p in ps:
    with open(p, encoding='utf-8') as f:
        data.append(f.read())
    fname = os.path.basename(p)
    sources.append(fname)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))


# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, DOCS_INDEX)
store.index = None
with open(FAISS_STORE_PKL, "wb") as f:
    pickle.dump(store, f)
