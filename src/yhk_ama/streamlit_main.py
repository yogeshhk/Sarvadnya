# Reference https://github.com/marshmellow77/streamlit-chatgpt-ui/blob/main/app.py
# https://github.com/hwchase17/langchain-streamlit-template/blob/master/main.py

# import openai
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import OpenAI, HuggingFaceHub
from PIL import Image
import pandas as pd
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent

# Model names
OPENAI = "GPT 3.5 (Open AI)"
FLAN = "Flan T5 (Google)"
DOLLY = "Dolly (Databricks)"

# Setting page title and header
st.set_page_config(page_title="AMA", page_icon=":teacher:")
st.markdown("<h1 style='text-align: center;'>AMA - Ask Me Anything chatbot</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []

if 'training_data' not in st.session_state:
    st.session_state['training_data'] = []

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
img = Image.open('logo.png')
st.sidebar.image(img, width=100)  # use_column_width=True
st.sidebar.title("Options")
training_files = st.sidebar.file_uploader("Training file", accept_multiple_files=True, type=['csv', 'txt', 'pdf'])
model_name = st.sidebar.radio("Choose LLM:", (OPENAI, FLAN, DOLLY))
clear_button = st.sidebar.button("Clear", key="clear")


def upload_training_data():
    training_data = []
    if training_files is not None:
        for file in training_files:
            df = pd.read_csv(file)
            training_data.append(df)
    return training_data


content = upload_training_data()
if len(content) > 0:
    st.session_state['training_data'] = content


def create_agent(llm_name):
    """Logic for loading the chain you want to use should go here."""
    llm = None
    agent_llm = None
    if llm_name == OPENAI:
        llm = OpenAI(temperature=0)
    elif llm_name == FLAN:
        llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 1e-10})
    elif llm_name == DOLLY:
        llm = HuggingFaceHub(repo_id="databricks/dolly-v2-3b", model_kwargs={"temperature": 0, "max_length": 64})

    if len(st.session_state["training_data"]) == 0:
        agent_llm = ConversationChain(llm=llm)
    else: # check if csv
        df = st.session_state["training_data"][0] # for now
        agent_llm = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=False)
    return agent_llm


agent = create_agent(model_name)

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['model_name'] = []
    st.session_state['training_data'] = []


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = agent.run(input=user_input)
    st.session_state['messages'].append({"role": "assistant", "content": response})
    return response


response_container = st.container()  # container for chat history
container = st.container()  # container for text box

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(f"Model used: {st.session_state['model_name'][i]}")
