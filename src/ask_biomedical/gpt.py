"""An NLP based search engine module to interact with open source knowledge graphs"""

import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

# os.environ['OPENAI_API_KEY'] = st.secrets["key"]


class GenerateCypher:
    def __init__(self, url:str, username:str, password:str) -> None:
        graph = Neo4jGraph(
            url=url, 
            username=username, 
            password=password
        )
        self.chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0),
            graph=graph,
            verbose=True,
        )

    def search(self, query: str) -> str:
        return self.chain.run(query)
    

def generate_response(prompt):
    url = st.secrets["url"]
    username = st.secrets["username"]
    password = st.secrets["password"]

    search_engine = GenerateCypher(url, username, password)
    message = search_engine.search(prompt)
    return message


def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text


