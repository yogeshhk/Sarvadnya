from os import path

BASE_DIR = path.dirname(__file__)

DOCS_INDEX = path.join(BASE_DIR, "models", "docs.index") # D:/Yogesh/Projects/SaaSGPT/Projects/AskAlmanack/models/docs.index
FAISS_STORE_PKL =  path.join(BASE_DIR, "models", "faiss_store.pkl") # "D:/Yogesh/Projects/SaaSGPT/Projects/AskAlmanack/models/faiss_store.pkl
DATA_FOLDER = path.join(BASE_DIR, "data") # "D:/Yogesh/Projects/SaaSGPT/Projects/AskAlmanack/data/"

APP_NAME = "Ask Almanack Bot"