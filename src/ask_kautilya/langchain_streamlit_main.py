"""Python file to serve as the frontend"""
import json
import os
import pickle

import faiss
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredHTMLLoader, TextLoader, PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceHubEmbeddings, LlamaCppEmbeddings
from langchain.llms import VertexAI, LlamaCpp
from langchain.vectorstores import FAISS
from streamlit_chat import message

from langchain.text_splitter import CharacterTextSplitter


class MyFAQsBot:
    def __init__(self, config_json):
        # print("in __init__")
        self.app_name = config_json['APP_NAME']
        self.files_paths = config_json['FILES_PATHS']
        self.docs_index = config_json['DOCS_INDEX']
        self.faiss_store_pkl = config_json['FAISS_STORE_PKL']
        self.model_name = config_json['MODEL_NAME']
        self.model_path = "./models/llama-7b.ggmlv3.q4_0.bin"  # hard coded

    def get_model(self):
        llm = None
        if self.model_name == "VertexAI":
            llm = VertexAI()  # need GCP account, project, own config set under ENV variable, refer README
        elif self.model_name == "Llama2":
            llm = LlamaCpp(model_path=self.model_path)
        return llm

    def generate_chain(self):
        if os.path.isfile(self.docs_index):
            # print("Just loading index as it is there")
            index = faiss.read_index(self.docs_index)
            with open(self.faiss_store_pkl, "rb") as f:
                store = pickle.load(f)
            store.index = index
        else:
            # print("populating index as it is not there")
            documents = []
            for p in self.files_paths:
                if p.lower().endswith('.csv'):
                    loader = CSVLoader(file_path=p)
                    documents += loader.load()
                elif p.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path=p)
                    documents += loader.load()
                elif p.lower().endswith('.txt'):
                    loader = TextLoader(file_path=p, encoding="UTF-8")
                    documents += loader.load()
                elif p.lower().endswith('.html'):
                    loader = UnstructuredHTMLLoader(file_path=p)
                    documents += loader.load()
                else:
                    st.write("Selected file extension not supported")

            # Split document in chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
            docs = text_splitter.split_documents(documents=documents)
            print(f"data has {len(docs)} documents")

            ## error for following code: ValueError: could not broadcast input array from shape (8,) into shape (0,)
            # if self.model_name == "Llama2":
            #     embeddings = LlamaCppEmbeddings(model_path=self.model_path)
            # else:
            embeddings = HuggingFaceHubEmbeddings()
            store = FAISS.from_documents(docs, embeddings)
            faiss.write_index(store.index, self.docs_index)
            with open(self.faiss_store_pkl, "wb") as f:
                pickle.dump(store, f)

        # print("Now index is there, create chain")
        db_as_retriver = store.as_retriever()
        llm = self.get_model()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=db_as_retriver, verbose=False, chain_type="stuff")
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

            result = chain(user_input)  # ({"question": user_input})

            st.session_state.past.append(user_input)
            st.session_state.generated.append(result['result'])

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def read_config(file_path="bot_config.json"):
    with open(file_path) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    config = read_config()
    bot = MyFAQsBot(config)
    bot.run_ui()
