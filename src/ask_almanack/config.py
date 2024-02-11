from os import path

BASE_DIR = path.dirname(__file__)

DOCS_INDEX = path.join(BASE_DIR, "vectorstore", "docs.index") # D:/Yogesh/GitHub/Sarvadnya/src/ask_almanack/vectorstore/docs.index
FAISS_STORE_PKL =  path.join(BASE_DIR, "vectorstore", "faiss_store.pkl") # "D:/Yogesh/GitHub/Sarvadnya/src/ask_almanack/vectorstore/faiss_store.pkl
DATA_FOLDER = path.join(BASE_DIR, "data") # "D:/Yogesh/GitHub/Sarvadnya/src/ask_almanack/data/"

APP_NAME = "Ask Almanack Bot"