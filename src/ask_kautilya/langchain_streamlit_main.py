# import json
# import os
# import pickle
# import faiss
# import streamlit as st
# from dotenv import load_dotenv

# from langchain.chains import RetrievalQA
# from langchain_community.document_loaders import (
#     UnstructuredHTMLLoader, TextLoader, PyPDFLoader, CSVLoader
# )
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import LlamaCpp
# from langchain_groq import ChatGroq
# from langchain.vectorstores import FAISS
# from langchain.text_splitter import CharacterTextSplitter
# from streamlit_chat import message

# # Load .env
# load_dotenv()

# GROQ_SUPPORTED_MODELS = [
    
#     "llama3-8b-8192",
#     "llama3-70b-8192"
# ]

# class MyFAQsBot:
#     def __init__(self, config_json):
#         self.app_name = config_json['APP_NAME']
#         self.files_paths = config_json['FILES_PATHS']
#         self.docs_index = config_json['DOCS_INDEX']
#         self.faiss_store_pkl = config_json['FAISS_STORE_PKL']
#         self.model_name = config_json['MODEL_NAME']
#         self.model_path = "./models/llama-7b.ggmlv3.q4_0.bin"

#     def get_model(self):
#         if self.model_name == "Groq":
#             selected_model = st.sidebar.selectbox("Select Groq Model", GROQ_SUPPORTED_MODELS, index=0)
#             return ChatGroq(temperature=0, model_name=selected_model)
#         elif self.model_name == "Llama2":
#             return LlamaCpp(model_path=self.model_path)
#         else:
#             raise ValueError(f"Unsupported model: {self.model_name}")

#     def generate_chain(self):
#         if os.path.isfile(self.docs_index) and os.path.isfile(self.faiss_store_pkl):
#             index = faiss.read_index(self.docs_index)
#             with open(self.faiss_store_pkl, "rb") as f:
#                 store = pickle.load(f)
#             store.index = index
#         else:
#             documents = []
#             for p in self.files_paths:
#                 ext = p.lower()
#                 if ext.endswith('.csv'):
#                     loader = CSVLoader(file_path=p)
#                 elif ext.endswith('.pdf'):
#                     loader = PyPDFLoader(file_path=p)
#                 elif ext.endswith('.txt'):
#                     loader = TextLoader(file_path=p, encoding="utf-8")
#                 elif ext.endswith('.html'):
#                     loader = UnstructuredHTMLLoader(file_path=p)
#                 else:
#                     st.warning(f"Unsupported file type: {p}")
#                     continue
#                 documents.extend(loader.load())

#             text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
#             docs = text_splitter.split_documents(documents)

#             embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 model_kwargs={"device": "cpu"},
#                 encode_kwargs={"normalize_embeddings": True}
#             )

#             store = FAISS.from_documents(docs, embeddings)

#             # Ensure model directory exists
#             os.makedirs(os.path.dirname(self.docs_index), exist_ok=True)

#             faiss.write_index(store.index, self.docs_index)
#             with open(self.faiss_store_pkl, "wb") as f:
#                 pickle.dump(store, f)

#         retriever = store.as_retriever()
#         llm = self.get_model()
#         return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", verbose=False)

#     def run_ui(self):
#         st.set_page_config(page_title=self.app_name, page_icon=":robot:")
#         st.header(self.app_name)

#         if "chain" not in st.session_state:
#             st.session_state["chain"] = self.generate_chain()

#         if "generated" not in st.session_state:
#             st.session_state["generated"] = []

#         if "past" not in st.session_state:
#             st.session_state["past"] = []

#         user_input = st.text_input("You: ", "", key="input")

#         if user_input:
#             chain = st.session_state["chain"]
#             result = chain.run(user_input)

#             st.session_state.past.append(user_input)
#             st.session_state.generated.append(result)

#         if st.session_state["generated"]:
#             for i in range(len(st.session_state["generated"]) - 1, -1, -1):
#                 message(st.session_state["generated"][i], key=str(i))
#                 message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


# def read_config(file_path="bot_config.json"):
#     with open(file_path) as f:
#         return json.load(f)


# if __name__ == "__main__":
#     config = read_config()
#     bot = MyFAQsBot(config)
#     bot.run_ui()
import json
import os
import pickle
import faiss
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    UnstructuredHTMLLoader, TextLoader, PyPDFLoader, CSVLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message

# Load .env
load_dotenv()
GROQ_SUPPORTED_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192"
]

class MyFAQsBot:
    def __init__(self, config_json):
        self.app_name = config_json['APP_NAME']
        self.files_paths = config_json['FILES_PATHS']
        self.docs_index = config_json['DOCS_INDEX']
        self.faiss_store_pkl = config_json['FAISS_STORE_PKL']
        self.model_name = config_json['MODEL_NAME']
        self.model_path = "./models/llama-7b.ggmlv3.q4_0.bin"

    def get_model(self):
        if self.model_name == "Groq":
            selected_model = st.sidebar.selectbox("Select Groq Model", GROQ_SUPPORTED_MODELS, index=0)
            return ChatGroq(temperature=0, model_name=selected_model)
        elif self.model_name == "Llama2":
            return LlamaCpp(model_path=self.model_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def generate_chain(self):
        if os.path.isfile(self.docs_index) and os.path.isfile(self.faiss_store_pkl):
            index = faiss.read_index(self.docs_index)
            with open(self.faiss_store_pkl, "rb") as f:
                store = pickle.load(f)
            store.index = index
        else:
            documents = []
            for p in self.files_paths:
                ext = p.lower()
                if ext.endswith('.csv'):
                    loader = CSVLoader(file_path=p)
                elif ext.endswith('.pdf'):
                    loader = PyPDFLoader(file_path=p)
                elif ext.endswith('.txt'):
                    loader = TextLoader(file_path=p, encoding="utf-8")
                elif ext.endswith('.html'):
                    loader = UnstructuredHTMLLoader(file_path=p)
                else:
                    st.warning(f"Unsupported file type: {p}")
                    continue
                documents.extend(loader.load())

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
            docs = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

            store = FAISS.from_documents(docs, embeddings)

            # Ensure model directory exists
            os.makedirs(os.path.dirname(self.docs_index), exist_ok=True)

            faiss.write_index(store.index, self.docs_index)
            with open(self.faiss_store_pkl, "wb") as f:
                pickle.dump(store, f)

        retriever = store.as_retriever()
        llm = self.get_model()
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", verbose=False)

    def run_ui(self):
        st.set_page_config(page_title=self.app_name, page_icon=":robot:")
        st.header(self.app_name)

        if "chain" not in st.session_state:
            st.session_state["chain"] = self.generate_chain()

        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        user_input = st.text_input("You: ", "", key="input")

        if user_input:
            chain = st.session_state["chain"]
            result = chain.run(user_input)

            st.session_state.past.append(user_input)
            st.session_state.generated.append(result)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def read_config(file_path="bot_config.json"):
    with open(file_path) as f:
        return json.load(f)


if __name__ == "__main__":
    config = read_config()
    bot = MyFAQsBot(config)
    bot.run_ui()
