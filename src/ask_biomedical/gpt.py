# """An NLP based search engine module to interact with open source knowledge graphs"""
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import GraphCypherQAChain
# # os.environ['OPENAI_API_KEY'] = st.secrets["key"]
# class GenerateCypher:
    # def __init__(self, url:str, username:str, password:str) -> None:

        # self.chain = GraphCypherQAChain.from_llm(
            # ChatOpenAI(temperature=0),
            # graph=graph,
            # verbose=True,
           # )
    # def search(self, query: str) -> str:
        # return self.chain.run(query)
               
import os
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from groq import Groq

# Set Groq API key
groq_api_key = os.getenv("YOUR-API-KEY") 
client = Groq(api_key=groq_api_key)

class GenerateCypher:
    def __init__(self, url: str, username: str, password: str) -> None:
        self.graph = Neo4jGraph(
            url=url,
            username=username,
            password=password
        )

    def generate_cypher_query(self, question: str) -> str:
        """
        Generate a Cypher query from the user's natural language question using Groq.
        """
        system_prompt = "You are an assistant that converts biomedical natural language questions into Cypher queries for a Neo4j graph database."
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Or llama3-70b-8192
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    def search(self, query: str) -> str:
        cypher = self.generate_cypher_query(query)
        try:
            result = self.graph.query(cypher)
            return f"Cypher: {cypher}\n\nResult:\n{result}"
        except Exception as e:
            return f"Cypher Query:\n{cypher}\n\nError running query: {str(e)}"

def generate_response(prompt):
    url = st.secrets["url"]
    username = st.secrets["username"]
    password = st.secrets["password"]

    search_engine = GenerateCypher(url, username, password)
    message = search_engine.search(prompt)
    return message

def get_text():
    return st.text_input("You:", "", key="input")
