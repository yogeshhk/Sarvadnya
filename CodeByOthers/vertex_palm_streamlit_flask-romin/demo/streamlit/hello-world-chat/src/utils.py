import streamlit as st
from streamlit_chat import message
import pandas as pd
import ast

def clear_chat() -> None:
    st.session_state['chat_model'] = ""
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['context'] = ""
    st.session_state['example'] = []
    st.session_state['temperature'] = []

def hard_reset_session() -> None: 
    st.session_state = {states : [] for states in st.session_state}


def create_session_state():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = []
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False
    if 'chat_input' not in st.session_state:
        st.session_state['chat_input'] = ''
    if 'context' not in st.session_state:
        st.session_state['context'] = ''
    if 'example' not in st.session_state:
        st.session_state['example'] = []
    if 'chat_model' not in st.session_state:
        st.session_state['chat_model'] = ""

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def download_chat():
    chat_data = pd.DataFrame([st.session_state['past'], st.session_state['generated']])
    csv = convert_df(chat_data)

    st.download_button(
    "Press to Download",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )

def chat_input_submit():
    st.session_state.chat_input = st.session_state.chat_widget
    st.session_state.chat_widget = ''

def clear_duplicate_data():
    for i in range(len(st.session_state['past'])-1,0,-1):
        if st.session_state['past'][i][i] == st.session_state['past'][i-1][i-1]:
            del st.session_state['past'][i]
            del st.session_state['generated'][i]

