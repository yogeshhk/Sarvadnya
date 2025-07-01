import json
import os
import pickle
from dotenv import load_dotenv

import faiss
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredHTMLLoader, TextLoader, PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import VertexAI, LlamaCpp
from langchain.vectorstores import FAISS
from streamlit_chat import message
from langchain.text_splitter import CharacterTextSplitter


class MyFAQsBot:
    def __init__(self, config_json):
        self.app_name = config_json['APP_NAME']
        self.files_paths = config_json['FILES_PATHS']
        self.docs_index = config_json['DOCS_INDEX']
        self.faiss_store_pkl = config_json['FAISS_STORE_PKL']
        self.model_name = config_json['MODEL_NAME']
        self.model_path = "./models/llama-7b.ggmlv3.q4_0.bin"

    def get_model(self):
        if self.model_name == "VertexAI":
            return VertexAI()
        elif self.model_name == "Llama2":
            return LlamaCpp(model_path=self.model_path)
        elif self.model_name == "Groq":
            from langchain_groq import ChatGroq
            return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def generate_chain(self):
        if os.path.isfile(self.docs_index):
            index = faiss.read_index(self.docs_index)
            with open(self.faiss_store_pkl, "rb") as f:
                store = pickle.load(f)
            store.index = index
        else:
            documents = []
            for p in self.files_paths:
                if p.lower().endswith('.csv'):
                    loader = CSVLoader(file_path=p)
                elif p.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path=p)
                elif p.lower().endswith('.txt'):
                    loader = TextLoader(file_path=p, encoding="UTF-8")
                elif p.lower().endswith('.html'):
                    loader = UnstructuredHTMLLoader(file_path=p)
                else:
                    st.write(f"Unsupported file type: {p}")
                    continue
                documents += loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
            docs = text_splitter.split_documents(documents=documents)
            print(f"data has {len(docs)} documents")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            store = FAISS.from_documents(docs, embeddings)
            # Ensure the models directory exists
            os.makedirs(os.path.dirname(self.docs_index), exist_ok=True)

# Write index
            faiss.write_index(store.index, self.docs_index)
            with open(self.faiss_store_pkl, "wb") as f:
                pickle.dump(store, f)

        db_as_retriever = store.as_retriever()
        llm = self.get_model()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=db_as_retriever, verbose=False, chain_type="stuff")
        return chain

    def run_ui(self):
        if "chain" not in st.session_state:
            st.session_state["chain"] = self.generate_chain()

        chain = st.session_state["chain"]

        st.set_page_config(page_title=self.app_name, page_icon=":robot:")
        st.header(self.app_name)

        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        user_input = st.text_input("You: ", "<type here>", key="input")
        prev_input = "<type here>"

        if prev_input != user_input:
            prev_input = user_input
            result = chain(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(result['result'])

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def read_config(file_path="bot_config.json"):
    load_dotenv()
    with open(file_path) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    config = read_config()
    bot = MyFAQsBot(config)
    bot.run_ui()
