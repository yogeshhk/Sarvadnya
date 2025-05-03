import os
import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_groq.chat_models import ChatGroq
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

## <YHK> Each client atomatically searches for these specific keys in ENV

# os.environ["TAVILY_API_KEY"]=st.secrets["TAVILY_API_KEY"]
# os.environ["GROQ_API_KEY"]=st.secrets["GROQ_API_KEY"]

# tavily_api_key = os.getenv("TAVILY_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")

#############################################################################
# 1. Define the GraphState 
#############################################################################
class GraphState(TypedDict):
    user_request: str
    requirements_docs_content: str
    requirements_docs_summary: str
    testcases_format: str
    testcases: str
    answer: str

#############################################################################
# 2. To generate Summary of the requirements document
#############################################################################
def generate_summary_node_function(state: GraphState) -> GraphState:
    """
    Uses LLM to generate summary of `requirements_docs_content`.
    """
    # print(f"YHK: inside generate_summary_node_function with state as {state}")

    requirements_docs_content = state.get("requirements_docs_content", "")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    prompt = (
    "You are an expert in generating QA testcases for any known formats. \n" + 
    "Study the given 'Requirements Documents Content' carefully and generate summary of about 5 lines\n" +
    f"Requirements Documents Content: {requirements_docs_content}\n" +
    "Answer:"
    )
    
    # print(f"YHK: inside generate_summary_node_function with prompt as:\n {prompt}")

    try:
        response = st.session_state.llm.invoke(prompt)
    except Exception as e:
        response = f"Error generating answer: {str(e)}"
        
    # print(f"YHK: returning from generate_summary_node_function with response as:\n {type(response)} also testcases {response.content}")
        
    state ['requirements_docs_summary'] = response.content
    state ['answer'] = response.content
    return state


#############################################################################
# 2. Router function to decide whether to output gherkin or selenium
#############################################################################
def route_user_request(state: GraphState) -> str:
    # print(f"YHK: inside route_user_request with state as {state}")
    # print(f"YHK: inside route_user_request with session state as {st.session_state}")

    user_request = state["user_request"]
    tool_selection = {
    "gherkin_format": (
        "Use requests generation of testcases in Gherkin format "
    ),
    "selenium_format": (
        "Use requests generation of testcases in Selenium format"
    )
    }

    SYS_PROMPT = """Act as a router to select specific testcase foramt or functions based on user's request, using the following rules:
                    - Analyze the given user's request and use the given tool selection dictionary to output the name of the relevant tool based on its description and relevancy. 
                    - The dictionary has tool names as keys and their descriptions as values. 
                    - Output only and only tool name, i.e., the exact key and nothing else with no explanations at all. 
                """

    # Define the ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            ("human", """Here is the user's request:
                        {user_request}
                        Here is the tool selection dictionary:
                        {tool_selection}
                        Output the required tool name from the tool selection dictionary only. Just one word.
                    """),
        ]
    )

    # Pass the inputs to the prompt
    inputs = {
        "user_request": user_request,
        "tool_selection": tool_selection
    }

    # Invoke the chain
    tool = (prompt | st.session_state.llm | StrOutputParser()).invoke(inputs)
    # print(f"YHK: route_user_request: raw {tool} output response")

    tool = re.sub(r"[\\'\"`]", "", tool.strip()) # Remove any backslashes and extra spaces
    # print(f"YHK: route_user_request: clean {tool} output response")
    
    ## <YHK> Assuming only 2 options for now
    if "gherkin" in tool:
        tool = "gherkin"
    else:
        tool = "selenium"
        
    state["testcases_format"] = tool
    
    print(f"YHK: returning from route_user_request with tool as: {tool}")
    return tool

def generate_testcases(user_request, requirements_content, llm, format_type):
    prompt = (
    "You are an expert in generating QA testcases for any known formats. \n" + 
    "Study the given 'Requirements Documents Content' carefully and generate about 3 testcases in the suggested 'Format'\n" +
    "You may want to look at the original User Request just to make sure that you are ansering th request properly.\n" +
    f"User Request: {user_request}\n" +
    f"Requirements Documents Content: {requirements_content}\n" +
    f"Format: {format_type}\n" +
    "Answer:"
    )
    
    # print(f"YHK: inside generate_testcases with prompt as:\n {prompt}")

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        response = f"Error generating answer: {str(e)}"
        
    # print(f"YHK: returning from generate_testcases with response as:\n {type(response)} also testcases {response.content}")
        
    return response.content

#############################################################################
# 3. To generate Gherikin formatted Testcases
#############################################################################
def generate_gherkin_testcases_node_function(state: GraphState) -> GraphState:
    """
    Uses LLM to generate Gherikin formatted Testcases of `requirements_docs_content`.
    """
    # print(f"YHK: inside generate_gherkin_testcases with state as {state}")

    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    testcases_format = state.get("testcases_format", "gherkin")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    response = generate_testcases(user_request, requirements_docs_content,st.session_state.llm, testcases_format)
    
    state ['testcases'] = response
    state ['answer'] = response
    return state


#############################################################################
# 4. To generate Selenium formatted Testcase
#############################################################################
def generate_selenium_testcases_node_function(state: GraphState) -> GraphState:
    """
    Uses LLM to generate Selenium formatted Testcases of `requirements_docs_summary`.
    """    
    # print(f"YHK: inside generate_selenium_testcases with state as {state}")
    
    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    testcases_format = state.get("testcases_format", "selenium")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    response = generate_testcases(user_request, requirements_docs_content,st.session_state.llm, testcases_format)
    
    state ['testcases'] = response
    state ['answer'] = response
    return state


#############################################################################
# 5. Build the LangGraph pipeline
#############################################################################
workflow = StateGraph(GraphState)
# Add nodes
workflow.add_node("summary_node", generate_summary_node_function)
workflow.add_node("gherkin_node", generate_gherkin_testcases_node_function)
workflow.add_node("selenium_node", generate_selenium_testcases_node_function)
# Start with summary node
# Then route from "route_user_request" to either "gherkin_node" or "selenium_node"
# From "gherkin_testcases" -> END
# From "geselenium_testcasesnerate" -> END 

# Add the Edges
# workflow.add_edge(START, "summary_node")
workflow.set_entry_point("summary_node")

workflow.add_conditional_edges(
    "summary_node",
    route_user_request,  # The router function, its output decides which node to go to
    {
        "gherkin": "gherkin_node",
        "selenium": "selenium_node"
    }
)
workflow.add_edge("gherkin_node", END)
workflow.add_edge("selenium_node", END)

#############################################################################
# 6. The initialize_app function
#############################################################################
def initialize_app(model_name: str):
    """
    Initialize the app with the given model name, avoiding redundant initialization.
    """
    # Check if the LLM is already initialized
    if "selected_model" in st.session_state and st.session_state.selected_model == model_name:
        return workflow.compile()  # Return the compiled workflow directly

    # Initialize the LLM for the first time or switch models
    st.session_state.llm = ChatGroq(model=model_name, temperature=0.0)
    st.session_state.selected_model = model_name
    print(f"Using model: {model_name}")
    print(f"Using state: {st.session_state.llm}")
    return workflow.compile()

