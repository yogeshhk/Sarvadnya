# Reference https://github.com/marshmellow77/streamlit-chatgpt-ui/blob/main/app.py
# https://github.com/hwchase17/langchain-streamlit-template/blob/master/main.py

# import openai
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import OpenAI, HuggingFaceHub
from PIL import Image

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


# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
img = Image.open('logo.png')
st.sidebar.image(img, width=100) #use_column_width=True)
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose LLM:", ("OpenAI", "Flan-T5"))
clear_button = st.sidebar.button("Clear", key="clear")


def load_chain(llm_name):
    """Logic for loading the chain you want to use should go here."""
    if llm_name == "OpenAI":
        llm = OpenAI(temperature=0)
    else:
        llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 1e-10})
    conversation_chain = ConversationChain(llm=llm)
    return conversation_chain


chain = load_chain(model_name)

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['model_name'] = []


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = chain.run(input=user_input)
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

