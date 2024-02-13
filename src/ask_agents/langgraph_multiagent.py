# Reference: https://levelup.gitconnected.com/langgraph-gemini-pro-custom-tool-streamlit-multi-agent-application-development-79c1473086b8
# https://www.youtube.com/watch?v=FWBnNcZv3kw

# Assuming VERTEX AI set in Environment variables
from langchain import hub
from langchain.agents import Tool, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph
from langchain_core.agents import AgentActionMessageLog
import streamlit as st

os.environ["SERPER_API_KEY"] = "YOUR-KEY-API" # TODO Change to duckduck go

search = GoogleSerperAPIWrapper()

st.set_page_config(page_title="LangChain Agent", layout="wide")


def main():
    # Streamlit UI elements
    st.title("LangGraph Agent + Gemini Pro + Custom Tool + Streamlit")

    # Input from user
    input_text = st.text_area("Enter your text:")

    if st.button("Run Agent"):

if __name__ == "__main__":
    main()
