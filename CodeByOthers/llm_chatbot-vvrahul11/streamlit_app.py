"""Module to create a chatbot interface, accept user input and generate output"""
import streamlit as st
from streamlit_chat import message


st.set_page_config(
    page_title="LLM Chatbot"
)
st.header("Biomedical LLM Chatbot")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''This is a web application that allows you to interact with an 
    EHR knowledge graph, ask biomedical questions or general questions. 
    '''
    )
st.sidebar.info('''Enter a query in the text box and press enter
    to receive a response''')

st.sidebar.info('''The app is under active development. 
    There are several issues that needs to be fix''')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


model = st.radio(
    "What task would you like to perform?",
    ('Biomedical KG question answering', 'Biomedical question answering', 'General question answering'))

if model == 'Biomedical KG question answering':
    st.text("This is a medical KG of drug, manufacturers and outcomes")
    from gpt import *
if model == 'Biomedical question answering':    
    from biogpt import *
if model =='General question answering':
    from falcon import *
    

user_input = get_text()

if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

